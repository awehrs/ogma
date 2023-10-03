from src.actor import Actor
from src.encoder import Encoder
from src.decoder import Decoder

from collections import deque
from dataclasses import dataclass
from typing import List, Mapping

from einops import rearrange
import jax.numpy as jnp
from jax import lax, random, vmap


class MemoryBuffer:
    """
    Circular buffer.

    Encoder L to R order: oldest activations -> newest activations.
    Decoder L to R order: nearest predictions -> most distant predictions.
    """

    def __init__(
        self, num_input_columns: int, input_dim: int, temporal_horizon: int = 2
    ):
        self.temporal_horizon = temporal_horizon
        enc_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns,), dtype="int16")
                for _ in range(temporal_horizon)
            ],
            maxlen=max(temporal_horizon, 1)
            # Bottom layer needs to be able to access current encoder activation.
        )
        dec_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns, input_dim), dtype="int16")
                for _ in range(temporal_horizon)
            ],
            maxlen=temporal_horizon + 1,
            # Need extra step of memory for decoder learning.
        )
        self.buffer = {"encoder": enc_buffer, "decoder": dec_buffer}

    def nearest(self, enc_or_dec: str) -> jnp.array:
        """Get closest activation in time; shape = (num_hidden_columns[, column_dim])"""
        if enc_or_dec == "encoder":
            return self.buffer[enc_or_dec][-1]
        else:
            return self.buffer[enc_or_dec][0]

    def _all(self, enc_or_dec: str) -> jnp.array:
        """Get all activations; shape = (temporal_horizon, num_hidden_columns[, column_dim])"""
        return jnp.concatenate(self.buffer[enc_or_dec], axis=0)

    def push(self, activation: jnp.array, enc_or_dec: str) -> None:
        if enc_or_dec == "encoder":
            self.buffer[enc_or_dec].append(activation)
        else:
            # Add multiple predictions at once.
            self.buffer[enc_or_dec].extend(
                jnp.split(activation, indices_or_sections=self.temporal_horizon)
            )


@dataclass
class Layer:
    temporal_horizon: int
    decoder: Decoder
    encoder: Encoder
    buffer: MemoryBuffer
    ticks_per_update: int
    ticks: int = 0
    updated: bool = False


class Network:
    """Sparse Predictive Heirarchy"""

    def __init__(
        self,
        layers: List[Layer],
        config: Mapping,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
    ):
        self.layers = layers
        self.config = config
        self.upward_mapping = upward_mapping
        self.downward_mapping = downward_mapping

    def step(self, precepts: jnp.array, learn: bool = True, act: bool = False):
        """Perform upward (encoder) and downward (decoder) pass."""
        # Upward pass.
        for l in range(len(self.layers)):
            layer = self.layers[l]

            if (l != 0) and (layer.ticks < layer.ticks_per_update):
                # Not ready to fire.
                continue

            # Encoder pass.
            inputs = precepts if l == 0 else layer.buffer._all("encoder")
            layer.ticks = 0
            layer.updated = True
            h_t = layer.encoder(
                input_activations=inputs,
                learn=learn,
                upward_mapping=self.upward_mapping,
                downward_mapping=self.downward_mapping,
            )

            # Update the weights of this layer's decoder.
            if learn:
                # Encoder output prediction was based on.
                prev_enc_output = layer.buffer.nearest("encoder")
                # Decoder output prediction was based on.
                next_layer_dec_output = (
                    self.layers[l + 2].buffer.nearest("decoder")
                    if l <= len(self.layers) - 1
                    else None
                )
                # Determine whether this was decoder's long or near term prediction.
                offset = (
                    1
                    if self.layers[l + 1].ticks < self.layers[l + 1].ticks_per_update
                    else 0
                )
                layer.decoder(
                    curr_target=h_t,
                    prev_prediction=layer.buffer.nearest("decoder"),
                    ctx_encoder=prev_enc_output,
                    ctx_decoder=next_layer_dec_output,
                    downward_mapping=self.downward_mapping,
                    offset=offset,
                    learn=True,
                )

            # Update encoder state after decoder learning,
            #  so that downward pass pops correct encoder output.
            layer.buffer.push(h_t, "encoder")
            if l < len(self.layers) - 1:
                self.layers[l + 1].ticks += 1

        # Downward pass.
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            if layer.updated:
                same_layer_enc_output = layer.buffer.nearest("encoder")
                next_layer_dec_output = (
                    self.layers[1 + 1].buffer.nearest("decoder")
                    if l < len(self.layers)
                    else None
                )
                y_t = layer.decoder(
                    ctx_encoder=same_layer_enc_output,
                    ctx_decoder=next_layer_dec_output,
                    downward_mapping=self.downward_mapping,
                    learn=False,
                    offset=5,
                )
                layer.buffer.push("decoder", y_t)
                layer.updated = False
        return y_t

    @classmethod
    def from_pretrained(config):
        raise NotImplementedError

    @classmethod
    def init_random(cls, config):
        key = random.PRNGKey(config.rng_seed)
        col_dim = config.hidden_column_dim
        layers = []
        # Build connections between layers.
        if config.y_dim is None:
            num_columns = config.x_dim
            receptive_area_up = 2 * config.up_radius + 1
            receptive_area_down = 2 * config.down_radius + 1
            up_mapping = cls.build_connection_mapping_1d(
                x_dim=config.x_dim, radius=config.up_radius, pad=config.pad
            )
            down_mapping = cls.build_connection_mapping_1d(
                x_dim=config.x_dim, radius=config.down_radius, pad=config.pad
            )
        else:
            num_columns = config.x_dim * config.y_dim
            receptive_area_up = (2 * config.up_radius + 1) ** 2
            receptive_area_down = (2 * config.down_radius + 1) ** 2
            up_mapping = cls.build_connection_mapping_2d(
                x_dim=config.x_dim,
                y_dim=config.y_dim,
                radius=config.up_radius,
                pad=config.pad,
            )
            down_mapping = cls.build_connection_mapping_2d(
                x_dim=config.x_dim,
                y_dim=config.y_dim,
                radius=config.down_radius,
                pad=config.pad,
            )
        # Build layers.
        for l in range(config.num_layers):
            if l == 0:
                # First layer has no encoder history.
                key, subkey = random.split(key)
                enc_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        col_dim,
                        receptive_area_up * config.preprocessor_dim,
                    ),
                )
                # But does have feedback.
                key, subkey = random.split(key)
                dec_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        config.temporal_horizon
                        * col_dim,  # Need to produce |temp_horizon| predictions.
                        2
                        * receptive_area_down
                        * col_dim,  # Feedback and enc_output get concat'ed, hence the 2.
                    ),
                )
                buf = MemoryBuffer(
                    num_input_columns=num_columns,
                    input_dim=col_dim,
                    temporal_horizon=config.temporal_horizon,
                )
            elif l == config.num_layers - 1:
                # Top layer has normal encoder horizon.
                key, subkey = random.split(key)
                enc_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        col_dim,
                        config.temporal_horizon * receptive_area_up * col_dim,
                    ),
                )
                # But no feeback (or memory buffer).
                key, subkey = random.split(key)
                dec_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        config.temporal_horizon
                        * col_dim,  # still produces multiple predictions
                        receptive_area_down
                        * col_dim,  # but don't have to concatentate feedback
                    ),
                )
            else:
                # Every layer in between.
                key, subkey = random.split(key)
                enc_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        col_dim,
                        config.temporal_horizon * receptive_area_up * col_dim,
                    ),
                )
                key, subkey = random.split(key)
                dec_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        config.temporal_horizon * col_dim,
                        2 * receptive_area_down * col_dim,
                    ),
                )
                buf = MemoryBuffer(
                    num_input_columns=num_columns,
                    input_dim=col_dim,
                    temporal_horizon=config.temporal_horizon,
                )
            enc = Encoder(
                parameters=enc_params,
                num_iters=config.num_iters,
                learning_rate=config.encoder_lr,
            )
            dec = Decoder(parameters=dec_params, learning_rate=config.decoder_lr)
            layers.append(
                Layer(
                    decoder=dec,
                    encoder=enc,
                    buffer=buf,
                    temporal_horizon=config.temporal_horizon,
                    ticks=0,
                    ticks_per_update=2**l,  # Exponential memory.
                )
            )
        return cls(
            layers=layers,
            config=config,
            upward_mapping=up_mapping,
            downward_mapping=down_mapping,
        )

    @classmethod
    def build_connection_mapping_1d(
        cls, x_dim: int, radius: int, pad: bool = False
    ) -> jnp.array:
        """
        For use with text, time series, and other 1d data.

        Building mapping from hidden column i's id, to the ids of
            the 2*r+1 input columns in i's receptive field.

        Returns array of size (num_hidden_columns, 2 * radius + 1)
        """
        assert x_dim >= 2 * radius + 1

        seq = jnp.arange(x_dim)

        if pad:
            # pad the sequence.
            idx = seq
            seq = jnp.pad(seq, pad_width=radius, mode="constant", constant_values=(-1,))
        else:
            # pad the idx.
            idx = jnp.arange(start=0, stop=x_dim - 2 * radius)
            idx = jnp.pad(idx, pad_width=radius, mode="edge")

        return vmap(lambda start: lax.dynamic_slice(seq, (start,), (2 * radius + 1,)))(
            idx
        )

    @classmethod
    def build_connection_mapping_2d(
        cls, x_dim: int, y_dim: int, radius: int, pad: bool = False
    ) -> jnp.array:
        """
        For use with images and other 2d data.

        Building mapping from hidden column i's id, to the ids of
            the 2*r+1 input columns in i's receptive field.

        Returns array of size (num_hidden_columns, 2 * radius + 1)
        """
        assert x_dim >= 2 * radius + 1
        assert y_dim >= 2 * radius + 1
        assert radius > 0

        matrix = jnp.arange(x_dim * y_dim)
        matrix = rearrange(matrix, "(x y) -> x y", x=x_dim)

        if pad:
            # Pad the sequence.
            idx = jnp.indices((x_dim, y_dim))
            matrix = jnp.pad(
                matrix,
                pad_width=radius,
                mode="constant",
                constant_values=(-1,),
            )
        else:
            # Pad the idx.
            idx = jnp.indices((x_dim - 2 * radius, y_dim - 2 * radius))
            idx = vmap(lambda m: jnp.pad(m, pad_width=radius, mode="edge"))(idx)

        idx_flat = rearrange(idx, "b x y -> b (x y)")
        idx_merged = vmap(lambda i, j: jnp.array([i, j]))(idx_flat[0], idx_flat[1])

        window_size = (2 * radius + 1, 2 * radius + 1)

        def map_fn(window_pos):
            i, j = window_pos
            window = lax.dynamic_slice(matrix, (i, j), window_size)
            return window

        windows = vmap(map_fn)(idx_merged)
        windows = rearrange(windows, "b w t -> b (w t)")

        return windows

    @property
    def num_params(self) -> int:
        params = 0
        for layer in self.layers:
            params += layer.enc.parameters.size
            params += layer.dec.parameters.size
        return params
