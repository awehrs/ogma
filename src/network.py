from src.actor import Actor
from src.encoder import Encoder, ReconstructionEncoder
from src.decoder import Decoder, LinearDecoder
from src.utils import dense_to_sparse, sparse_to_dense

from collections import deque
from dataclasses import dataclass
import logging
from typing import Callable, List, Mapping, Tuple

from einops import rearrange, repeat
import jax.numpy as jnp
from jax import lax, random, vmap

logger = logging.getLogger()


class MemoryBuffer:
    """
    Stores encoder activations (k-hot) and decoder predictions (dense).

    Encoder L to R order: oldest activations -> newest activations.
    Decoder L to R order: nearest predictions -> most distant predictions.
    """

    def __init__(
        self,
        k_hot: int,
        decoder_dim: int,
        num_input_columns: int,
        temporal_horizon: int,
        num_decoder_predictions: int,
    ):
        self.temporal_horizon = temporal_horizon
        self.num_decoder_predictions = num_decoder_predictions

        enc_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns, k_hot), dtype="int16")
                for _ in range(temporal_horizon)
            ],
            maxlen=temporal_horizon,
        )

        dec_buffer = deque(
            [
                jnp.zeros(shape=(num_input_columns, decoder_dim), dtype=jnp.float32)
                for _ in range(num_decoder_predictions)
            ],
            maxlen=num_decoder_predictions,
        )

        self.buffer = {"encoder": enc_buffer, "decoder": dec_buffer}

    def nearest(self, enc_or_dec: str) -> jnp.array:
        """Get closest activation in time; shape = (num_hidden_columns, k_hot/column_dim)"""
        if enc_or_dec == "encoder":
            return self.buffer[enc_or_dec][-1]
        else:
            return self.buffer[enc_or_dec][0]

    def _all(self, enc_or_dec: str) -> deque[jnp.array]:
        """Get all activations; shape = (temporal_horizon, num_hidden_columns, k_hot/column_dim])"""
        return self.buffer[enc_or_dec]

    def pop(self, enc_or_dec: str) -> None:
        if enc_or_dec == "encoder":
            raise NotImplementedError
        else:
            self.buffer["decoder"].popleft()

    def push(self, activation: jnp.array, enc_or_dec: str) -> None:
        assert len(activation.shape) > 1

        if enc_or_dec == "encoder":
            self.buffer[enc_or_dec].append(activation)
        else:
            # Add multiple predictions at once.
            self.buffer[enc_or_dec].extend(
                jnp.split(
                    activation,
                    indices_or_sections=self.num_decoder_predictions,
                    axis=-1,
                )
            )

    @property
    def decoder(self) -> deque[jnp.array]:
        return self.buffer["decoder"]

    @property
    def encoder(self) -> deque[jnp.array]:
        return self.buffer["encoder"]


@dataclass
class Layer:
    level: int
    temporal_horizon: int
    decoder_params: jnp.array
    encoder_params: jnp.array
    decoder_losses: List[float]
    encoder_losses: List[float]
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
        decoder: Decoder,
        encoder: Encoder,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
    ):
        self.layers = layers
        self.config = config
        self.upward_mapping = upward_mapping
        self.downward_mapping = downward_mapping

        self.decoder_loss_fn = self.build_loss_fn(config.loss_fn)
        self.encoder_loss_fn = self.build_loss_fn(config.loss_fn)
        self.decoder_forward_fn = self.build_decoder_forward_fn(
            decoder, config.async_step
        )
        self.decoder_learn_fn = self.build_decoder_learn_fn(decoder, config.async_step)
        self.encoder_step_fn = self.build_encoder_step_fn(encoder, config.async_step)

    def step(self, precepts: jnp.array, learn: bool = True, act: bool = False):
        """Perform upward (decoder-learning, encoder) and downward (decoder) pass."""

        # Upward pass.
        for layer in self.layers:
            layer.ticks += 1

            if layer.ticks < layer.ticks_per_update:
                # Layer not ready to fire.
                continue

            inputs = self.get_layer_inputs(layer=layer, precepts=precepts)

            # Update the weights of this layer's decoder.
            if learn:
                context = self.build_decoder_context(layer=layer, learn=learn)
                offset, prediction, target = self.build_decoder_example(
                    layer=layer, inputs=inputs
                )
                layer.decoder_params = self.decoder_learn_fn(
                    context=context,
                    prediction=prediction,
                    target=target,
                    downward_mapping=self.downward_mapping,
                    parameters=layer.decoder_params,
                    offset=offset,
                    learning_rate=self.config.decoder_lr,
                )

            # Encoder pass.
            input_col_dim = self.get_input_column_dim(layer.level)

            parameters, output, reconstruction = self.encoder_step_fn(
                input_activations=inputs,
                parameters=layer.encoder_params,
                upward_mapping=self.upward_mapping,
                downward_mapping=self.downward_mapping,
                k_hot=self.config.k_hot,
                learn=learn,
                input_column_dim=input_col_dim,
                learning_rate=self.config.encoder_lr,
                num_iters=self.config.num_iters,
            )

            # Update encoder state.
            if learn:
                layer.encoder_params = parameters

            layer.ticks = 0
            layer.updated = True
            layer.buffer.push(output, "encoder")

        # Downward pass.
        for layer in reversed(self.layers):
            if not layer.updated:
                # Layer not ready to activate.
                continue

            context = self.build_decoder_context(layer=layer, learn=False)

            y_t = self.decoder_forward_fn(
                context=context,
                parameters=layer.decoder_params,
                downward_mapping=self.downward_mapping,
                k_hot=self.config.k_hot,
            )

            # Update decoder buffer.
            layer.buffer.push(y_t, "decoder")
            layer.updated = False

    def naive_async_step(self, precepts: jnp.array, learn: bool):
        layers = jnp.arange(start=0, stop=self.config.num_layers)
        inputs = [
            self.get_layer_inputs(precepts=precepts, layer=l) for l in self.layers
        ]
        context = [
            self.build_decoder_context(layer=l, learn=False) for l in self.layers
        ]
        example = [
            self.build_decoder_example(layer=l, inputs=i)
            for (i, l) in zip(inputs, self.layers)
        ]
        import time

        f = vmap(
            lambda ctxt: self.layers[1].decoder.forward(
                ctxt,
                self.layers[1].decoder_params,
                self.downward_mapping,
                self.config.k_hot,
            )
        )
        f(jnp.stack(context[:-1]))
        t1 = time.time()
        f(jnp.stack(context[:-1]))
        t2 = time.time()

        print(t2 - t1)
        assert False
        params = vmap(lambda layer, input: "layer.decoder.forward(input)")(
            "layers", inputs
        )
        for l in self.layers:
            l.decoder_params = params[l.layer]

        params, output, recons = vmap(lambda layer, input: "layer.encoder.step(input)")(
            "layers", inputs
        )
        inputs = "async gather_decoder_forward_inputs(layers)"
        prediction = vmap(lambda layer, inputs: "layer.decoder.forward(input)")(
            "layers", inputs
        )

    def build_decoder_context(self, layer: Layer, learn: bool) -> jnp.array:
        """
        Combine (same layer) encoder output with decoder feeback, to create
            context for decoder to consume.
        """
        # Get most recent output of layer's encoder.
        enc_output = self.get_layer_output(layer=layer, learn=learn)

        # Get feedback from next layer's decoder.
        feedback = self.get_feedback(layer=layer, learn=learn)

        if layer.level == len(self.layers) - 1:
            # shape = [num_columns, k_hot/hidden_dim]
            context = enc_output
        else:
            # Shape = [num_columns, 2 * k_hot/hidden_column_dimension]:
            context = jnp.concatenate([feedback, enc_output], axis=-1)
            if not learn:
                context = self.adjust_dimensions(
                    context,
                    column_dimension=self.config.hidden_column_dim,
                    k_hot=self.config.k_hot,
                )

        return context

    def build_decoder_example(
        self, layer: Layer, inputs: jnp.array
    ) -> Tuple[jnp.array]:
        """
        Arrange (target, prediction) example for decoder learning.
        """
        # Get the previous prediction of this layer's decoder.
        prev_prediction = self.get_prev_prediction(layer)

        # Get offset index for parameter updating.
        offset = layer.buffer.decoder.maxlen - len(layer.buffer.decoder)
        layer.buffer.pop("decoder")

        # Get the target of the prediction.
        targets = self.get_targets(layer=layer, inputs=inputs)

        return offset, prev_prediction, targets

    def get_input_column_dim(self, layer_num: int) -> int:
        if layer_num == 0:
            return self.config.preprocessor_dim
        else:
            return self.config.hidden_column_dim * self.config.temporal_horizon

    def get_layer_inputs(self, layer: Layer, precepts: jnp.array) -> jnp.array:
        """
        Get the feedforward inputs to a given layer.

        Returns arry of shape (num_columns, temporal_horizon, k_hot).
        """
        if layer.level == 0:
            # Bottom layer has no history.
            inputs = precepts
        else:
            # Every other layer pulls from a memory buffer.
            inputs = self.layers[layer.level - 1].buffer._all("encoder")
            inputs = jnp.concatenate(inputs, axis=-1)
            inputs = self.adjust_dimensions(
                inputs,
                column_dimension=self.config.hidden_column_dim,
                k_hot=self.config.k_hot,
            )

        return inputs

    def get_layer_output(self, layer: Layer, learn: bool) -> jnp.array:
        """Get layer's most recent encoder ouput."""
        enc_output = layer.buffer.nearest("encoder")

        if learn:
            enc_output = sparse_to_dense(
                enc_output, dim=self.config.hidden_column_dim, k_hot=self.config.k_hot
            )

        return enc_output

    def get_feedback(self, layer: Layer, learn: bool) -> jnp.array:
        """Get the feedback from next layer's decoder."""
        if layer.level == len(self.layers) - 1:
            # Top layer receives no feedback.
            feedback = None
        else:
            feedback = self.layers[layer.level + 1].buffer.nearest("decoder")
            if not learn:
                feedback = dense_to_sparse(feedback, k_hot=self.config.k_hot)

        return feedback

    def get_prev_prediction(self, layer: Layer) -> jnp.array:
        """Get the prediction made by a decoder during its previous activation."""

        return layer.buffer.nearest("decoder")

    def get_targets(self, layer: Layer, inputs: jnp.array) -> jnp.array:
        """Build the values a decoder prediction is targeting."""
        if layer.level == 0:
            targets = sparse_to_dense(
                inputs,
                dim=self.config.preprocessor_dim,
                k_hot=self.config.k_hot,
            )
        else:
            targets = sparse_to_dense(
                self.layers[layer.level - 1].buffer.nearest("encoder"),
                dim=self.config.hidden_column_dim,
                k_hot=self.config.k_hot,
            )

        return targets

    def get_prediction(self):
        """Get prediction of next input."""
        layer = self.layers[0]
        dense_pred = layer.buffer.nearest("decoder")
        sparse_pred = dense_to_sparse(dense_pred, k_hot=self.config.k_hot)

        return sparse_pred

    @classmethod
    def from_pretrained(config):
        raise NotImplementedError

    @classmethod
    def init_random(cls, config):
        if config.async_step:
            assert config.hidden_column_dim == config.preprocessor_dim

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

        # Build encoders/decoders.
        decoder = cls.init_decoder(config.decoder_type)
        encoder = cls.init_encoder(config.encoder_type)

        # Build layers.
        for l in range(config.num_layers):
            # Build encoder.
            key, subkey = random.split(key)
            enc_params = cls.init_encoder_params(layer_num=l, key=subkey, config=config)

            # Build decoder.
            key, subkey = random.split(key)
            dec_params = cls.init_decoder_params(layer_num=l, key=subkey, config=config)

            # Build memory buffer.
            buf = cls.init_memory_buffer(layer_num=l, config=config)

            # Add layer.
            layers.append(
                Layer(
                    level=l,
                    decoder_params=dec_params,
                    encoder_params=enc_params,
                    decoder_losses=[],
                    encoder_losses=[],
                    buffer=buf,
                    temporal_horizon=config.temporal_horizon if l != 0 else 1,
                    ticks=0,
                    ticks_per_update=2**l,  # Exponential memory.
                )
            )

        # Adjust parameters to allow for layerwise parallelism.
        if config.async_step:
            layers = cls.pad_encoder_params(layers)
            layers = cls.pad_decoder_params(layers)

        return cls(
            layers=layers,
            config=config,
            decoder=decoder,
            encoder=encoder,
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
            k_hot=config.k_hot,
            decoder_dim=decoder_dim,
            num_input_columns=cls.num_columns(config),
            temporal_horizon=config.temporal_horizon,
            num_decoder_predictions=num_preds,
        )

    @staticmethod
    def pad_encoder_params(layers: List[Layer]) -> List[Layer]:
        standard_dim = layers[-1].encoder_params.shape[-1]
        bottom_layer_dim = layers[0].encoder_params.shape[-1]

        layers[0].encoder_params = jnp.pad(
            layers[0].encoder_params,
            pad_width=((0, 0), (0, 0), (0, standard_dim - bottom_layer_dim)),
            constant_values=0,
        )

        assert all(
            layer.encoder_params.shape == layers[0].encoder_params.shape
            for layer in layers
        )

        return layers

    @staticmethod
    def pad_decoder_params(layers: List[Layer]) -> List[Layer]:
        standard_output_dim = layers[-1].decoder_params.shape[1]
        standard_input_dim = layers[0].decoder_params.shape[-1]
        bottom_layer_output_dim = layers[0].decoder_params.shape[1]
        top_layer_input_dim = layers[-1].decoder_params.shape[-1]

        layers[0].decoder_params = jnp.pad(
            layers[0].decoder_params,
            pad_width=(
                (0, 0),
                (0, standard_output_dim - bottom_layer_output_dim),
                (0, 0),
            ),
            constant_values=0,
        )

        layers[-1].decoder_params = jnp.pad(
            layers[-1].decoder_params,
            pad_width=(
                (0, 0),
                (0, 0),
                (0, standard_input_dim - top_layer_input_dim),
            ),
            constant_values=0,
        )

        return layers

    @staticmethod
    def adjust_dimensions(
        columns: jnp.array, column_dimension: int, k_hot: int
    ) -> jnp.array:
        """
        Args:
            columns: activation array of shape (num_hidden_columns, k_hot * num_vecs)
            column_dimension: dense dimension of vectors that were concatenated to form 'columns'
            k_hot: number of active cells per vector that got concatenated
        """
        num_vecs = columns.shape[-1] // k_hot
        offset = jnp.arange(
            start=0, stop=num_vecs * column_dimension, step=column_dimension
        )
        offset = repeat(offset, "d -> (d k)", k=k_hot)
        return columns + offset

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

    @staticmethod
    def init_encoder(encoder_type: str) -> Encoder:
        if encoder_type == "reconstruction":
            return ReconstructionEncoder()
        else:
            raise NotImplementedError

    @staticmethod
    def init_decoder(decoder_type: str) -> Decoder:
        if decoder_type == "linear":
            return LinearDecoder()
        else:
            raise NotImplementedError

    @staticmethod
    def build_decoder_forward_fn(decoder: Decoder, async_step: bool):
        if not async_step:
            return decoder.forward

        if isinstance(decoder, LinearDecoder):
            forward_fn = vmap(
                lambda context, parameters, downward_mapping, k_hot: decoder.forward(
                    context, parameters, downward_mapping, k_hot
                ),
                in_axes=(0, 0, None, None),
            )
        else:
            raise NotImplementedError

        return forward_fn

    @staticmethod
    def build_decoder_learn_fn(decoder: Decoder, async_step: bool):
        if not async_step:
            return decoder.learn

        if isinstance(decoder, LinearDecoder):
            learn_fn = vmap(
                lambda context, prediction, target, downward_mapping, parameters, offset, learning_rate: decoder.learn(
                    context,
                    prediction,
                    target,
                    parameters,
                    offset,
                    downward_mapping,
                    learning_rate,
                ),
                in_axes=(0, 0, 0, 0, 0, None, None),
            )
        else:
            raise NotImplementedError

        return learn_fn

    @staticmethod
    def build_encoder_step_fn(encoder: Encoder, async_step: bool):
        if not async_step:
            return encoder.step

        if isinstance(encoder, ReconstructionEncoder):
            learn_fn = vmap(
                lambda input_activations, parameters, upward_mapping, downward_mapping, k_hot, learn, input_column_dim, learning_rate, num_iters: encoder.step(
                    input_activations,
                    parameters,
                    upward_mapping,
                    downward_mapping,
                    k_hot,
                    learn,
                    input_column_dim,
                    learning_rate,
                    num_iters,
                ),
                in_axes=(0, 0, None, None, None, None, None, None, None),
            )
        else:
            raise NotImplementedError

        return learn_fn

    @staticmethod
    def build_loss_fn(loss_fn: str) -> Callable:
        pass

    @property
    def num_params(self) -> int:
        params = 0
        for layer in self.layers:
            params += layer.encoder_params.size
            params += layer.decoder_params.size

        # TODO
        if self.config.async_step:
            # Subtract dummy parameters.
            pass

        return params
