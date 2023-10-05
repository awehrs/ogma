from src.propagate import propagate

from typing import Tuple

from jax import nn, vmap
import jax.numpy as jnp


class Decoder:
    """Sparse decoder."""

    def __init__(self, parameters: jnp.array, learning_rate: int):
        self.parameters = parameters
        self.lr = learning_rate
        self.optimizer = None

    def activate(self, h: jnp.array) -> jnp.array:
        """One hot activation."""
        return jnp.argmax(h, axis=1)

    def forward(self, input: jnp.array, parameters: jnp.array) -> jnp.array:
        h = propagate(input, parameters, input_is_one_hot=True, output_is_one_hot=False)
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
        input_losses = jnp.expand_dims(input_losses, axis=1)
        previous_context = jnp.expand_dims(previous_context, axis=1)
        return jnp.kron(input_losses, jnp.transpose(previous_context))

    def _learn(
        self,
        target: jnp.array,
        prediction: jnp.array,
        context: jnp.array,
        parameters: jnp.array,
        learning_rate: float,
    ) -> Tuple[jnp.array]:
        loss = self.loss(prediction, target)

        delta = vmap(self.update)(loss, context)

        parameters += learning_rate * delta

        return loss, parameters

    def learn(
        self,
        target: jnp.array,
        prediction: jnp.array,
        context: jnp.array,
    ):
        """Outter learn function."""
        loss, parameters = self._learn(
            target=target,
            prediction=prediction,
            context=context,
            parameters=self.parameters,
            learning_rate=self.lr,
        )

        self.parameters = parameters

        return loss, parameters

    def _step(
        self,
        context: jnp.array,
        parameters: jnp.array,
    ) -> Tuple[jnp.array]:
        return self.forward(context, parameters)

    def step(
        self,
        context: jnp.array,
    ) -> Tuple[jnp.array]:
        return self._step(
            context=context,
            parameters=self.parameters,
        )
