from src.propagate import propagate
from src.utils import stride_inputs

from abc import ABC, abstractmethod
from functools import partial

from einops import rearrange
from jax import lax, nn, jit, vmap
import jax.numpy as jnp


class Decoder(ABC):
    """Abstract base class for stateless decoder."""

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def loss():
        pass

    @abstractmethod
    def update():
        pass

    @abstractmethod
    def learn():
        pass


class LinearDecoder(Decoder):
    """Sparse linear decoder."""

    @staticmethod
    @partial(
        jit,
        static_argnames=[
            "k_hot",
        ],
    )
    def forward(
        context: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        k_hot: int,
    ) -> jnp.array:
        """
        Args:
            input: array of shape (num_columns, n * k_hot),
                where n depends on the layer level.
            parameters: array of shape (num_columns, output_dim, recepetive_area * hidden_dim).
            downward_mapping: array of shape (num_columns, receptive_area)
            k_hot: number of active cells per input column.
        Returns:
            array of shape (num_columns, output_dim)
        """
        context = stride_inputs(inputs=context, input_mapping=downward_mapping)
        h = propagate(
            context,
            parameters,
            k_hot_input=k_hot,
            k_hot_output=None,
        )
        return nn.softmax(h)

    @staticmethod
    @partial(
        jit,
        static_argnames=[
            "offset",
            "learning_rate",
        ],
    )
    def learn(
        context: jnp.array,
        prediction: jnp.array,
        target: jnp.array,
        parameters: jnp.array,
        offset: int,
        downward_mapping: jnp.array,
        learning_rate: float,
        is_async: bool = False,
    ) -> jnp.array:
        """
        Args:
            context: array of shape (num_columns, context_dim),
                where 'n' depends on layer level.
            prediction: array of shape (num_columns, output_dim).
            target: array of shape (num_columns, output_dim).
            downward_mapping: array of shape (num_columns, receptive_area).
            parameters: array of shape
                (num_columns, num_predictions * output_dim, receptive_area * context_dim).
            offset: index offset for parameter updates.
            learning_rate: factor by which to incorporate updates.
        Returns:
            parameters: updated parameters, or NoneType if learn == False.
            output: array of shape (num_columns, k_hot).
        """
        loss = LinearDecoder.loss(prediction, target)

        loss = stride_inputs(inputs=loss, input_mapping=downward_mapping)

        delta = vmap(LinearDecoder.update)(loss, context)

        if isinstance(offset, jnp.ndarray):
            parameters.at[:, offset, :].add(learning_rate * delta)
        else:
            parameters.at[:, offset : offset + prediction.shape[1], :].add(
                learning_rate * delta
            )

        return parameters

    def loss(prediction: jnp.array, target: jnp.array) -> jnp.array:
        """ "
        Args:
            prediction: array of shape (num_columns, output_dim).
            target: array of shape (num_columns, output_dim).
        """
        return target - prediction

    def update(
        input_losses: jnp.array,
        previous_context: jnp.array,
    ) -> jnp.array:
        """
        Args:
            input_losses: array of shape (receptive_area, context_dim)
            previous_context: array of shape (prediction_dim,)
        Returns:
            array of shape (receptive_area, prediction_dim, context_dim)
        """
        input_losses = rearrange(input_losses, "r d -> d r")
        previous_context = jnp.expand_dims(previous_context, axis=1)
        return jnp.kron(input_losses, jnp.transpose(previous_context))
