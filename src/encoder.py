from src.utils import compressed_to_full, stride_inputs
from src.propagate import propagate

from typing import Tuple

from einops import rearrange
from jax import vmap
import jax.numpy as jnp


class Encoder:
    """Exponential Reconstruction Encoder."""

    def __init__(self, parameters: jnp.array, num_iters: int, learning_rate: int):
        self.parameters = parameters
        self.num_iters = num_iters
        self.lr = learning_rate
        self.optimizer = None

    def forward(
        self,
        input_activiations: jnp.array,
        parameters: jnp.array,
        input_is_one_hot: bool,
        output_is_one_hot: bool,
    ) -> jnp.array:
        return propagate(
            input_activiations,
            parameters,
            input_is_one_hot=input_is_one_hot,
            output_is_one_hot=output_is_one_hot,
        )

    def backward(
        self,
        output_activations: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        output_is_one_hot: bool,
    ) -> jnp.array:
        output_activations = stride_inputs(output_activations, downward_mapping)
        parameters = rearrange(
            parameters, "n h (r i) -> n i (r h)", r=output_activations.shape[-1]
        )
        recons = propagate(
            output_activations,
            parameters,
            input_is_one_hot=True,
            output_is_one_hot=output_is_one_hot,
        )
        recons = recons / max(1, downward_mapping.shape[1])
        return jnp.exp(jnp.minimum(recons - 1, jnp.zeros_like(recons)))

    def learn(
        self,
        input_activations: jnp.array,
        parameters: jnp.array,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
        num_iters: int,
        learning_rate: float,
    ) -> jnp.array:
        input_col_dim = parameters.shape[1]
        hidden = compressed_to_full(
            jnp.zeros_like(input_activations), dim=input_col_dim
        )
        recons = jnp.zeros_like(input_activations)

        for i in range(num_iters):
            inputs = stride_inputs(input_activations - recons, upward_mapping)
            hidden += self.forward(
                inputs,
                parameters,
                input_is_one_hot=(i < 1),
                output_is_one_hot=False,
            )
            recons = self.backward(
                jnp.argmax(hidden, axis=1),
                parameters,
                downward_mapping,
                output_is_one_hot=False,
            )
            if i == 0:
                input_activations = compressed_to_full(
                    input_activations, dim=input_col_dim
                )

        loss = self.loss(input_activations, recons)

        loss = stride_inputs(loss, upward_mapping)

        hidden = jnp.argmax(hidden, axis=1)

        hidden = compressed_to_full(hidden, dim=input_col_dim)

        delta = vmap(self.update)(loss, hidden)

        parameters += learning_rate * delta

        return parameters

    def loss(self, input_column: jnp.array, recons_column: jnp.array) -> jnp.array:
        """
        Calculate reconstruction loss of a single input column. Isolated here
            to allow vmap'ing.
        """
        return input_column - recons_column

    def update(
        self,
        input_losses: jnp.array,
        hidden_column: jnp.array,
    ) -> jnp.array:
        """
        Args:
            input_losses: array of shape (receptive_area, input_dim)
            hidden_column: array of shape (hidden_dim,)

        Returns:
            array of shape (receptive_area, hidden_dim, input_dim)
        """
        return jnp.kron(jnp.transpose(input_losses), hidden_column)

    def step(
        self,
        input_activations: jnp.array,
        parameters: jnp.array,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
        num_iters: int,
        learning_rate: float,
        learn: bool = True,
    ) -> Tuple[jnp.array]:
        if learn:
            parameters = self.learn(
                input_activations,
                parameters,
                upward_mapping,
                downward_mapping,
                num_iters,
                learning_rate,
            )
        return self.forward(
            stride_inputs(input_activations, upward_mapping),
            parameters,
            input_is_one_hot=True,
            output_is_one_hot=True,
        )

    def __call__(
        self,
        input_activations: jnp.array,
        learn: bool,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
    ):
        if len(input_activations.shape) > 1:
            # vmap/pmap step over batch dimension
            pass

        return self.step(
            input_activations,
            parameters=self.parameters,
            num_iters=self.num_iters,
            upward_mapping=upward_mapping,
            downward_mapping=downward_mapping,
            learning_rate=self.lr,
            learn=learn,
        )
