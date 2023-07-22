from src.propagate import propagate

from typing import Any, Tuple

from einops import rearrange
from jax import grad, lax, vmap
import jax.numpy as jnp


class Decoder:
    """Logistic regression decoder."""

    def __init__(self, parameters: jnp.array, learning_rate: int):
        self.parameters = parameters
        self.lr = learning_rate
        self.optimizer = None

    def activate(self, h: jnp.array) -> jnp.array:
        return lax.logistic(h)

    def forward(self, context: jnp.array, parameters: jnp.array) -> jnp.array:
        h = propagate(
            context, parameters, input_is_one_hot=True, output_is_one_hot=False
        )
        return (jnp.argmax(h, axis=1),)

    def loss(self, prediction: jnp.array, input: jnp.array):
        return jnp.mean((prediction - input) ** 2)

    def update(self):
        def kronecker(loss_col):
            return jnp.kron(loss_col, hidden_column)

        update = vmap(kronecker)(input_losses)
        update = rearrange(update, "r i h -> r h i")
        return update

    def learn(
        self,
        prev_prediction: jnp.array,
        current_input: jnp.array,
        parameters: jnp.array,
        learning_rate: float,
    ) -> jnp.array:
        x = grad(self.loss)(prev_prediction, current_input)
        parameters += learning_rate * x

    def step(
        self,
        inputs: jnp.array,
        prev_prediction: jnp.array,
        current_input: jnp.array,
        parameters: jnp.array,
        learning_rate: float,
        learn: bool = True,
    ) -> Tuple[jnp.array]:
        if learn:
            parameters = self.learn(
                prev_prediction, current_input, parameters, learning_rate
            )
        y = self.forward(inputs, parameters)
        y_t_2n, y_t_4n = jnp.split(y, indices_or_sections=2, axis=0)
        return y_t_2n, y_t_4n

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
