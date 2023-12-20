from src.actor import Actor
from src.encoder import Encoder
from src.decoder import Decoder
from src.utils import sparse_to_dense, stride_inputs

from collections import deque
from dataclasses import dataclass
import logging
from typing import List, Mapping

from einops import rearrange
import jax.numpy as jnp
from jax import lax, random, vmap

logger = logging.getLogger()


class MemoryBuffer:
    """
    Stores encoder activations (one hot) and decoder predictions (dense).

    Encoder L to R order: oldest activations -> newest activations.
    Decoder L to R order: nearest predictions -> most distant predictions.
    """

    def __init__(
        self,
        decoder_dim: int,
        num_input_columns: int,
        temporal_horizon: int,
        num_decoder_predictions: int,
    ):
        self.temporal_horizon = temporal_horizon
        self.num_decoder_predictions = num_decoder_predictions

        enc_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns,), dtype="int16")
                for _ in range(temporal_horizon)
            ],
            maxlen=temporal_horizon,
        )

        dec_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns, decoder_dim), dtype=jnp.float32)
                for _ in range(num_decoder_predictions + 1)
            ],
            maxlen=num_decoder_predictions + 1,
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
                jnp.split(activation, indices_or_sections=self.num_decoder_predictions)
            )

    @property
    def decoder_buffer(self) -> deque[jnp.array]:
        return self.buffer["decoder"]

    @property
    def encoder_buffer(self) -> deque[jnp.array]:
        return self.buffer["encoder"]


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
        print("Upward pass")
        print("///")
        for layer in self.layers:
            layer.ticks += 1

            if layer.ticks < layer.ticks_per_update:
                # Layer not ready to fire.
                continue

            print("Layer:", layer.level)

            inputs = self.get_layer_inputs(layer=layer, precepts=precepts)

            # Update the weights of this layer's decoder.
            if learn:
                print("decoder learning")
                loss = self.decoder_call(layer=layer, inputs=inputs, learn=True)

                if self.config.log_losses:
                    logger.info(f"Layer number {layer.level} decoder loss: {loss}")

            # Encoder pass.
            print("encoder pass")
            print("///")
            h_t = layer.encoder(
                input_activations=inputs,
                learn=learn,
                upward_mapping=self.upward_mapping,
                downward_mapping=self.downward_mapping,
            )

            # Update encoder state.
            layer.ticks = 0
            layer.updated = True
            layer.buffer.push(h_t, "encoder")

        # Downward pass.
        print("downward pass")
        for layer in reversed(self.layers):
            if not layer.updated:
                # Layer not ready to activate.
                continue

            print("layer: ", layer.level)
            y_t = self.decoder_call(layer=layer, inputs=inputs, learn=False)

            # Update decoder buffer.
            layer.buffer.push(y_t, "decoder")
            layer.updated = False

    def decoder_call(self, layer: Layer, inputs: jnp.array, learn: bool) -> None:
        """
        Gather a layer's previous context and, possibly, prediction and targets;
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
            inputs = jnp.concatenate(inputs, axis=0)

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
                self.layers[layer.level - 1].buffer.nearest("encoder"),
                dim=self.config.hidden_column_dim,
            )

        return targets

    def get_prediction(self):
        """Get prediction of next input."""
        layer = self.layers[0]
        dense_pred = layer.buffer.nearest("decoder")
        sparse_pred = layer.decoder.activate(dense_pred)

        return sparse_pred

    @classmethod
    def from_pretrained(config):
        raise NotImplementedError

    @classmethod
    def init_random(cls, config):
        key = random.PRNGKey(config.rng_seed)
        layers = []
        # Build connections between layers.
        if config.y_dim is None:
            up_mapping = cls.build_connection_mapping_1d(
                x_dim=config.x_dim, radius=config.up_radius, pad=config.pad
            )
            down_mapping = cls.build_connection_mapping_1d(
                x_dim=config.x_dim, radius=config.down_radius, pad=config.pad
            )
        else:
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
            # Build encoder.
            key, subkey = random.split(key)
            enc_params = cls.init_encoder_params(layer_num=l, key=subkey, config=config)
            enc = Encoder(
                parameters=enc_params,
                num_iters=config.num_iters,
                learning_rate=config.encoder_lr,
            )
            # Build decoder.
            key, subkey = random.split(key)
            dec_params = cls.init_decoder_params(layer_num=l, key=subkey, config=config)
            dec = Decoder(parameters=dec_params, learning_rate=config.decoder_lr)

            # Build memory buffer.
            buf = cls.init_memory_buffer(layer_num=l, config=config)

            # Add layer.
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
        assert radius > 0
        assert x_dim >= 2 * radius + 1
        assert y_dim >= 2 * radius + 1

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

    @classmethod
    def init_encoder_params(
        cls, key: random.PRNGKey, layer_num: int, config: Mapping
    ) -> jnp.array:
        num_columns = cls.num_columns(config)
        output_dim = config.hidden_column_dim
        receptive_area_up = cls.receptive_area(config, direction="up")

        if layer_num == 0:
            input_dim = receptive_area_up * config.preprocessor_dim
        else:
            input_dim = (
                config.temporal_horizon * receptive_area_up * config.hidden_column_dim
            )

        return random.normal(
            key,
            shape=(
                num_columns,
                output_dim,
                input_dim,
            ),
        )

    @classmethod
    def init_decoder_params(
        cls, key: random.PRNGKey, layer_num: int, config: Mapping
    ) -> jnp.array:
        num_columns = cls.num_columns(config)
        output_dim = config.hidden_column_dim
        receptive_area_down = cls.receptive_area(config, direction="down")
        num_decoder_predictions = cls.num_decoder_predictions(
            layer_num, schedule_type=config.schedule_type
        )

        if layer_num == 0:
            # Feedback and enc_output get concat'ed, hence the 2.
            input_dim = 2 * receptive_area_down * config.hidden_column_dim
            output_dim = config.preprocessor_dim
        elif layer_num == config.num_layers - 1:
            # Don't have to concatentate feedback
            input_dim = receptive_area_down * config.hidden_column_dim
            output_dim = num_decoder_predictions * config.hidden_column_dim
        else:
            input_dim = 2 * receptive_area_down * config.hidden_column_dim
            output_dim = num_decoder_predictions * config.hidden_column_dim

        return random.normal(
            key,
            shape=(
                num_columns,
                output_dim,
                input_dim,
            ),
        )

    @classmethod
    def init_memory_buffer(cls, layer_num: int, config: Mapping) -> MemoryBuffer:
        decoder_dim = (
            config.preprocessor_dim if layer_num == 0 else config.hidden_column_dim
        )
        num_preds = cls.num_decoder_predictions(layer_num, config.schedule_type)

        return MemoryBuffer(
            decoder_dim=decoder_dim,
            num_input_columns=cls.num_columns(config),
            temporal_horizon=config.temporal_horizon,
            num_decoder_predictions=num_preds,
        )

    @staticmethod
    def num_columns(config: Mapping) -> int:
        if config.y_dim is None:
            return config.x_dim
        else:
            return config.x_dim * config.y_dim

    @staticmethod
    def receptive_area(config: Mapping, direction: str) -> int:
        diameter = (
            (2 * config.up_radius) + 1
            if direction == "up"
            else (2 * config.down_radius) + 1
        )
        if config.y_dim is None:
            return diameter
        else:
            return diameter**2

    @staticmethod
    def num_decoder_predictions(layer_num: int, schedule_type: str) -> int:
        if schedule_type == "exponential":
            num_pred = 2**layer_num // 2 ** (layer_num - 1) if layer_num > 0 else 1
            return num_pred
        else:
            raise NotImplementedError

    @property
    def num_params(self) -> int:
        params = 0
        for layer in self.layers:
            params += layer.encoder.parameters.size
            params += layer.decoder.parameters.size
        return params
