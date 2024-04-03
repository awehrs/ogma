from src.actor import Actor
from src.encoder import Encoder, ReconstructionEncoder
from src.decoder import Decoder, LinearDecoder
from src.utils import dense_to_sparse, sparse_to_dense

from collections import deque
from dataclasses import dataclass
import logging
from typing import Callable, List, Mapping, Optional, Tuple

from einops import rearrange, repeat
import jax.numpy as jnp
from jax import lax, random, jit, vmap

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
    decoder_padding: Mapping[int, int]
    encoder_padding: Mapping[int, int]
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
        if self.config.async_step:
            self.async_step(precepts, learn)
        else:
            self.sync_step(precepts, learn)

    def sync_step(
        self, precepts: jnp.array, learn: bool = True, act: bool = False
    ) -> None:
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
                dec_params = layer.decoder_params = self.decoder_learn_fn(
                    context=context,
                    prediction=prediction,
                    target=target,
                    downward_mapping=self.downward_mapping,
                    parameters=layer.decoder_params,
                    offset=offset,
                    learning_rate=self.config.decoder_lr,
                )
            else:
                dec_params = offset = prediction = target = None

            # Encoder pass.
            input_col_dim = self.get_input_column_dim(layer.level)

            enc_params, enc_output, reconstruction = self.encoder_step_fn(
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

            if learn:
                layer.decoder_params = dec_params
                layer.encoder_params = enc_params

            layer.ticks = 0
            layer.updated = True
            layer.buffer.push(enc_output, "encoder")
            # self.update_layer_state(
            #     layer,
            #     learn=learn,
            #     up_or_down="up",
            #     async_step=False,
            #     decoder_params=dec_params,
            #     encoder_params=enc_params,
            #     encoder_output=enc_output,
            #     input_reconstruction=reconstruction,
            #     target=target,
            #     prediction=prediction,
            # )

        # Downward pass.
        for layer in reversed(self.layers):
            if not layer.updated:
                # Layer not ready to activate.
                continue

            context = self.build_decoder_context(layer=layer, learn=False)

            dec_output = self.decoder_forward_fn(
                context=context,
                parameters=layer.decoder_params,
                downward_mapping=self.downward_mapping,
                k_hot=self.config.k_hot,
            )

            layer.udpated = False
            layer.buffer.push(dec_output, "decoder")
            # self.update_layer_state(
            #     layer,
            #     learn=False,
            #     up_or_down="down",
            #     async_step=False,
            #     decoder_output=dec_output,
            # )

    def async_step(self, precepts: jnp.array, learn: bool):
        """
        Step all ready-to-fire layers in parallel.
        """
        # Determine which layers are firing.
        import time

        t1 = time.time()
        active_layers = []
        for layer in self.layers:
            layer.ticks += 1
            if layer.ticks >= layer.ticks_per_update:
                active_layers.append(layer)
        print("active layers:", len(active_layers))
        # Gather params.
        enc_params = jnp.stack([layer.encoder_params for layer in active_layers])
        dec_params = jnp.stack([layer.decoder_params for layer in active_layers])

        # Gather inputs.
        layer_inputs = [
            self.get_layer_inputs(layer, precepts) for layer in active_layers
        ]
        inputs_array = jnp.stack(layer_inputs)

        # Do decoder learning.
        if learn:
            # Gather decoder inputs.
            layer_context = [
                self.build_decoder_context(layer, learn) for layer in active_layers
            ]
            context_array = jnp.stack(layer_context)
            layer_example = [
                self.build_decoder_example(active_layers[i], layer_inputs[i])
                for i in range(len(active_layers))
            ]
            offset_array = jnp.stack(
                [
                    jnp.arange(start=example[0], stop=example[1].shape[1])
                    for example in layer_example
                ]
            )
            prediction_array = jnp.stack([example[1] for example in layer_example])
            target_array = jnp.stack([example[2] for example in layer_example])
            t2 = time.time()
            print("gathering inputs took:", t2 - t1)
            t1 = time.time()
            # Layer parallel decoder learn.
            dec_params = self.decoder_learn_fn(
                context_array,
                prediction_array,
                target_array,
                dec_params,
                offset_array,
                self.downward_mapping,
                self.config.decoder_lr,
            )

        # Do encoder step
        input_column_dim = self.get_input_column_dim(layer_num=1)

        enc_params, enc_output, reconstruction = self.encoder_step_fn(
            inputs_array,
            enc_params,
            self.upward_mapping,
            self.downward_mapping,
            self.config.k_hot,
            learn,
            input_column_dim,
            self.config.encoder_lr,
            self.config.num_iters,
        )
        t2 = time.time()
        print("up pass took:", t2 - t1)
        t1 = time.time()
        # Update state.
        for i in range(len(enc_params)):
            active_layers[i].updated = True
            active_layers[i].ticks = 0
            active_layers[i].decoder_params = dec_params[i, :, :, :]
            active_layers[i].encoder_params = enc_params[i, :, :, :]
            active_layers[i].buffer.push(
                enc_output[
                    i,
                    :,
                    :,
                ],
                "encoder",
            )
            eval_metric = reconstruction

        # Gather inputs again.
        layer_context = [
            self.build_decoder_context(layer, learn=False) for layer in active_layers
        ]
        context_array = jnp.stack(layer_context)
        t2 = time.time()
        print("updating up state, gathering down inputs took:", t2 - t1)
        # Decoder foward.
        t1 = time.time()
        dec_output = self.decoder_forward_fn(
            context_array,
            dec_params,
            self.downward_mapping,
            self.config.k_hot,
        )
        t2 = time.time()
        print("down pass took:", t2 - t1)
        # Update state.
        t1 = time.time()
        for i in range(len(active_layers)):
            eval_metric = None
            active_layers[i].updated = False
            if i == 0:
                active_layers[i].buffer.push(
                    dec_output[i, :, : self.config.hidden_column_dim], "decoder"
                )
            else:
                active_layers[i].buffer.push(dec_output[i, :, :], "decoder")
        t2 = time.time()
        print("updating down state took:", t2 - t1)

    def update_layer_state(
        self,
        layer: Layer,
        learn: bool,
        up_or_down: str,
        async_step: bool = False,
        decoder_params: Optional[jnp.array] = None,
        encoder_params: Optional[jnp.array] = None,
        decoder_output: Optional[jnp.array] = None,
        encoder_output: Optional[jnp.array] = None,
        input_reconstruction: Optional[jnp.array] = None,
        target: Optional[jnp.array] = None,
        prediction: Optional[jnp.array] = None,
    ) -> None:
        # Update after upward pass.
        if up_or_down == "up":
            if learn:
                layer.decoder_params = decoder_params
                layer.encoder_params = encoder_params
                # layer.decoder_losses.append(
                #     self.decoder_loss_fn(target, prediction)
                # )
                # layer.encoder_losses.append(
                #     self.encoder_loss_fn(
                #         encoder_output, input_reconstruction
                #     )
                # )
            layer.ticks = 0
            layer.updated = True
            layer.buffer.push(encoder_output, "encoder")
        # Update after downward pass
        else:
            layer.udpated = False
            layer.buffer.push(decoder_output, "decoder")

    def build_decoder_context(self, layer: Layer, learn: bool) -> jnp.array:
        """
        Combine (same layer) encoder output with decoder feeback, to create
            context for decoder to consume.
        """
        # Get most recent output of layer's encoder.
        enc_output = self.get_layer_output(layer=layer, learn=learn)

        # Get feedback from next layer's decoder.
        feedback = self.get_feedback(layer=layer, learn=learn)

        if layer.level == len(self.layers) - 1 and not self.config.async_step:
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

        Args:
            precepts: array of shape (k_hot,)

        Returns arry of shape (num_columns, temporal_horizon, k_hot).
        """
        if layer.level == 0:
            # Bottom layer has no history.
            if not self.config.async_step:
                return precepts
            else:
                inputs = jnp.pad(
                    precepts,
                    pad_width=((0, 0), (0, self.config.temporal_horizon - 1)),
                    constant_values=self.config.temporal_horizon
                    * self.config.hidden_column_dim
                    + 1,
                    # Ensures out of bounds indexing, and hence matmul eval to 0.
                    # See propagate and adjust_dimensions
                )
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
            if self.config.async_step:
                feedback = jnp.full(
                    shape=(self.upward_mapping.shape[0], self.config.hidden_column_dim),
                    fill_value=2 * self.config.hidden_column_dim + 1,
                )  # Ensures out of bounds indexing, and zero filling
            else:
                return None
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
        if layer.level == 0 and not self.config.async_step:
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

    def num_padded_steps(self, layer: Layer, up_or_down: str) -> int:
        return layer.encoder_padding[2] // (
            self.config.hidden_column_dim * self.receptive_area(self.config, up_or_down)
        )

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
                    decoder_padding={},
                    encoder_padding={},
                    decoder_losses=[],
                    encoder_losses=[],
                    buffer=buf,
                    temporal_horizon=(
                        config.temporal_horizon
                        if (l != 0) or (config.async_step)
                        else 1
                    ),
                    ticks=0,
                    ticks_per_update=2**l,  # Exponential memory.
                )
            )

        # Adjust parameters to allow for layerwise parallelism.
        # if config.async_step:
        #     layers = cls.pad_encoder_params(layers, config)
        #     layers = cls.pad_decoder_params(layers, config)

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

        if layer_num == 0 and not config.async_step:
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
        if config.async_step and layer_num == 0:
            layer_num += 1
        num_columns = cls.num_columns(config)
        receptive_area_down = cls.receptive_area(config, direction="down")
        num_decoder_predictions = cls.num_decoder_predictions(
            layer_num, schedule_type=config.schedule_type
        )
        if layer_num == 0:
            # Feedback and enc_output get concat'ed, hence the 2.
            input_dim = 2 * receptive_area_down * config.hidden_column_dim
            output_dim = config.preprocessor_dim
        elif layer_num == config.num_layers - 1 and not config.async_step:
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
                lambda context, prediction, target, parameters, offset, downward_mapping, learning_rate: decoder.learn(
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
