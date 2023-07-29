from src.propagate import propagate
from utils import stride_inputs

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
        inputs: jnp.array,
        prev_prediction: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        learning_rate: float,
        learn: bool = True,
    ) -> Tuple[jnp.array]:
        if learn:
            parameters = self.learn(inputs, prev_prediction, parameters, learning_rate)
        y = self.forward(inputs, parameters)
        y_t_2n, y_t_4n = jnp.split(y, indices_or_sections=2, axis=0)
        return y_t_2n, y_t_4n

    def __call__(
        self,
        inputs: jnp.array,
        prev_prediction: jnp.array,
        downward_mapping: jnp.array,
        learn: bool,
    ) -> Any:
        return self.step(
            inputs=inputs,
            prev_prediction=prev_prediction,
            parameters=self.parameters,
            downward_mapping=downward_mapping,
            learning_rate=self.lr,
            learn=learn,
        )
