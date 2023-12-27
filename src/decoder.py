from src.propagate import propagate
from src.utils import stride_inputs

from typing import Optional, Tuple

from einops import rearrange
from jax import nn, vmap
import jax.numpy as jnp


class Decoder:
    """Sparse decoder."""

    def __init__(self, parameters: jnp.array, learning_rate: int):
        self.parameters = parameters
        self.lr = learning_rate
        self.optimizer = None

    def forward(self, input: jnp.array, parameters: jnp.array, k_hot: int) -> jnp.array:
        h = propagate(
            input,
            parameters,
            k_hot_input=k_hot,
            k_hot_output=None,
        )
        return nn.softmax(h)

    def loss(self, prediction: jnp.array, target: jnp.array):
        return target - prediction

    def update(
        self,
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

    def learn(
        self,
        context: jnp.array,
        prediction: jnp.array,
        target: jnp.array,
        downward_mapping: jnp.array,
        offset: int,
        parameters: jnp.array,
        learning_rate: float,
    ) -> Tuple[jnp.array]:
        loss = self.loss(prediction, target)

        loss = stride_inputs(inputs=loss, input_mapping=downward_mapping)

        delta = vmap(self.update)(loss, context)

        parameters.at[:, offset : offset + prediction.shape[1], :].add(
            learning_rate * delta
        )

        return parameters

    def __call__(
        self,
        context: jnp.array,
        prediction: Optional[jnp.array] = None,
        target: Optional[jnp.array] = None,
        downward_mapping: Optional[jnp.array] = None,
        offset: Optional[int] = None,
        k_hot: Optional[int] = None,
    ) -> jnp.array:
        if target is not None:
            # Learn mode.
            parameters = self.learn(
                context=context,
                prediction=prediction,
                target=target,
                downward_mapping=downward_mapping,
                offset=offset,
                parameters=self.parameters,
                learning_rate=self.lr,
            )
            self.parameters = parameters
            return prediction, target
        else:
            # Forward mode.
            context = stride_inputs(inputs=context, input_mapping=downward_mapping)
            return self.forward(input=context, parameters=self.parameters, k_hot=k_hot)
