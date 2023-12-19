from src.actor import Actor
from src.encoder import Encoder
from src.decoder import Decoder
from src.utils import sparse_to_dense, stride_inputs

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
            [jnp.zeros(shape=(num_input_columns,), dtype="int16")],
            maxlen=temporal_horizon,
        )
        dec_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns, input_dim), dtype="int16")
                for _ in range(temporal_horizon + 1)
            ],
            maxlen=temporal_horizon + 1,
        )
        self.buffer = {"encoder": enc_buffer, "decoder": dec_buffer}

    def nearest(self, enc_or_dec: str) -> jnp.array:
        """Get closest activation in time; shape = (num_hidden_columns[, column_dim])"""
        if enc_or_dec == "encoder":
            return self.buffer[enc_or_dec][-1]
        else:
            return self.buffer[enc_or_dec][0]

    def _all(self, enc_or_dec: str) -> deque[jnp.array]:
        """Get all activations; shape = (temporal_horizon, num_hidden_columns[, column_dim])"""
        return self.buffer[enc_or_dec]

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
    level: int
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
        """Perform upward (decoder-learning, encoder) and downward (decoder) pass."""

        # Upward pass.
        for layer in self.layers:
            if (layer.level != 0) and (layer.ticks < layer.ticks_per_update):
                # Layer not ready to fire.
                continue

            inputs = self.get_layer_inputs(layer=layer, precepts=precepts)

            # Update the weights of this layer's decoder.
            if learn:
                self.decoder_call(layer=layer, inputs=inputs, learn=True)

            # Encoder pass.
            layer.ticks = 0
            layer.updated = True

            h_t = layer.encoder(
                input_activations=inputs,
                learn=learn,
                upward_mapping=self.upward_mapping,
                downward_mapping=self.downward_mapping,
            )

            # Update encoder state.
            layer.buffer.push(h_t, "encoder")

        # Downward pass.
        for layer in reversed(self.layers):
            if not layer.updated:
                # Layer not ready to activate.
                continue

            y_t = self.decoder_call(layer=layer, inputs=inputs, learn=False)

            # Update decoder buffer.
            layer.buffer.push(y_t, "decoder")

            layer.updated = False

        return y_t

    def decoder_call(self, layer: Layer, inputs: jnp.array, learn: bool) -> None:
        """
        Gather a layer's previous context and, possibly, prediction, and targets;
            and pass them to decoder's learning or forward function.
        """
        # Get most recent output of layer's encoder.
        enc_output = self.get_layer_output(layer=layer, learn=learn)

        # Get feedback from next layer's decoder.
        feedback = self.get_feedback(layer=layer, learn=learn)

        # Build context the decoder conditioned its previous prediction on.
        context = self.build_decoder_context(
            layer=layer,
            feedback=feedback,
            enc_output=enc_output,
        )

        # Get the previous prediction of this layer's decoder.
        prev_prediction = self.get_prev_prediction(layer) if learn else None

        # Get the target of the prediction.
        targets = self.get_targets(layer=layer, inputs=inputs) if learn else None

        return layer.decoder(
            context=context,
            prediction=prev_prediction,
            target=targets,
        )

    def get_offset(self, layer: Layer) -> int:
        """
        Convenience method. Determines if layer is firing at the same time as
            the layer above it (offset = 1) or  in between the activations of
            the layer above it (offset = 0).
        """
        offset = (
            0
            if self.layers[layer.level + 1].ticks
            < self.layers[layer.level + 1].ticks_per_update
            else 1
        )
        return offset

    def get_layer_output(self, layer: Layer, learn: bool) -> jnp.array:
        """Get layer's most recent encoder ouput."""
        enc_output = layer.buffer.nearest("encoder")

        if learn:
            enc_output = sparse_to_dense(enc_output, dim=self.config.hidden_column_dim)

        return enc_output

    def get_layer_inputs(self, layer: Layer, precepts: jnp.array) -> jnp.array:
        """Get the feedforward inputs to a given layer."""
        if layer.level == 0:
            # Bottom layer has no history.
            inputs = precepts
        else:
            # Every other layer pulls from a memory buffer.
            inputs = self.layers[layer.level - 1].buffer._all("encoder")
            inputs = jnp.stack(inputs, axis=1)
            # Shape = [num_columns, receptive_area * 2], that look like:
            # [[inputs[0:r]_t-1 | inputs[0:r]_t], [inputs[r:2r]_t-1 | inputs[r:2r]_t], ...]
            inputs = rearrange(inputs, "n (d r) -> (n r) d")

        return inputs

    def get_feedback(self, layer: Layer, learn: bool) -> jnp.array:
        """Get the feedback from next layer's decoder."""
        if layer.level == len(self.layers):
            # Top layer receives no feedback.
            feedback = None
        else:
            offset = self.get_offset(layer)
            feedback = self.layers[layer.level + 1].buffer._all("decoder")[offset]
            if not learn:
                feedback = layer.decoder.activate(feedback)

        return feedback

    def build_decoder_context(
        self, layer: Layer, feedback: jnp.array, enc_output: jnp.array
    ) -> jnp.array:
        """
        Combine (same layer) encoder output with decoder feeback, to create
            context for decoder to consume.
        """
        if layer.level == len(self.layers):
            # Shape = [num_columns, receptive_area]
            context = stride_inputs(enc_output, self.downward_mapping)
        else:
            if len(feedback.shape) == 2:  # Dense inputs.
                # Shape = [num_columns, 2]:
                context = jnp.concatenate([feedback, enc_output], axis=-1)
            else:  # Sparse inputs
                # Shape = [num_columns, 2]:
                context = jnp.stack([feedback, enc_output], axis=1)
            # Shape = [num_columns, recepetive_area, 2]:
            context = stride_inputs(context, self.downward_mapping)
            # Shape = [num_columns, receptive_area * 2]:
            context = rearrange(context, "n r d -> n (d r)")

        return context

    def get_prev_prediction(self, layer: Layer) -> jnp.array:
        """Get the prediction made by a decoder during its previous activation."""
        if layer.level == 0:
            prev_prediction = layer.buffer.nearest("decoder")
        else:
            prev_prediction = layer.buffer._all("decoder")
            # Shape = [num_columns, 2 * column_dim]:
            prev_prediction = jnp.concatenate(
                [prev_prediction[1], prev_prediction[2]], axis=-1
            )
            prev_prediction = rearrange(prev_prediction, "n (x d) -> n x d", x=2)

        return prev_prediction

    def get_targets(self, layer: Layer, inputs: jnp.array) -> jnp.array:
        """Build the values a decoder prediction is targeting."""
        if layer.level == 0:
            targets = sparse_to_dense(
                inputs,
                dim=self.config.preprocessor_dim,
            )
        else:
            targets = sparse_to_dense(
                self.layers[layer - 1].buffer.nearest("encoder"),
                dim=self.config.hidden_column_dim,
            )

        return targets

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
                # And only makes one prediction.
                key, subkey = random.split(key)
                dec_params = random.normal(
                    subkey,
                    shape=(
                        num_columns,
                        col_dim,
                        2 * receptive_area_down * col_dim,
                    ),  # Feedback and enc_output get concat'ed, hence the 2.
                )
                buf = MemoryBuffer(
                    num_input_columns=num_columns,
                    input_dim=col_dim,
                    temporal_horizon=1,
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
                # But receives no feedback.
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
                    level=l,
                    decoder=dec,
                    encoder=enc,
                    buffer=buf,
                    temporal_horizon=config.temporal_horizon if l != 0 else 1,
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
            params += layer.encoder.parameters.size
            params += layer.decoder.parameters.size
        return params
