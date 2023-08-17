from src.propagate import propagate
from utils import compressed_to_full, stride_inputs

from typing import Any, Tuple

from einops import rearrange
from jax import grad, nn, vmap
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

    def loss(self, prediction: jnp.array, inputs: jnp.array):
        """Cross-entropy loss."""
        return -jnp.sum(inputs * jnp.log(prediction))

    def learn(
        self,
        inputs: jnp.array,
        prev_prediction: jnp.array,
        parameters: jnp.array,
        learning_rate: float,
    ) -> jnp.array:
        x = grad(self.loss)(prev_prediction, inputs)
        parameters += learning_rate * x

    def step(
        self,
        feedforward: jnp.array,
        feedback: jnp.array,
        prev_prediction: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        learning_rate: float,
        learn: bool = True,
    ) -> Tuple[jnp.array]:
        if learn:
            parameters = self.learn(
                inputs=compressed_to_full(feedforward, dim=prev_prediction.shape[-1]),
                prev_prediction=prev_prediction,
                parameters=parameters,
                learning_rate=learning_rate,
            )
        inputs = stride_inputs(
            jnp.concatenate([feedforward, feedback], axis=0), downward_mapping
        )
        return self.forward(inputs, parameters)

    def __call__(
        self,
        feedforward: jnp.array,
        feedback: jnp.array,
        prev_prediction: jnp.array,
        downward_mapping: jnp.array,
        learn: bool,
    ) -> Any:
        return self.step(
            feedforward=feedforward,
            feedback=feedback,
            prev_prediction=prev_prediction,
            parameters=self.parameters,
            downward_mapping=downward_mapping,
            learning_rate=self.lr,
            learn=learn,
        )
