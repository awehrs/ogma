from src.utils import sparse_to_dense, stride_inputs
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
        input_col_dim = int(parameters.shape[-1] / upward_mapping.shape[-1])
        hidden_col_dim = parameters.shape[1]

        activation = jnp.zeros(shape=(downward_mapping.shape[0], parameters.shape[1]))
        recons = jnp.zeros_like(input_activations)

        for i in range(num_iters):
            inputs = stride_inputs(input_activations - recons, upward_mapping)
            activation += self.forward(
                inputs,
                parameters,
                input_is_one_hot=(i < 1),
                output_is_one_hot=False,
            )
            recons = self.backward(
                jnp.argmax(activation, axis=1),
                parameters,
                downward_mapping,
                output_is_one_hot=False,
            )
            if i == 0:
                input_activations = sparse_to_dense(
                    input_activations, dim=input_col_dim
                )

        loss = self.loss(input_activations, recons)

        hidden = jnp.argmax(activation, axis=1)

        hidden = sparse_to_dense(hidden, dim=hidden_col_dim)

        hidden = stride_inputs(hidden, downward_mapping)

        hidden = rearrange(hidden, "n r h -> n (r h)")

        update = vmap(self.update, in_axes=(0, 0, None))
        delta = update(
            loss,
            hidden,
            upward_mapping.shape[-1],
        )

        parameters += learning_rate * delta

        return parameters

    def loss(self, inputs: jnp.array, reconstruction: jnp.array) -> jnp.array:
        return inputs - reconstruction

    def update(
        self,
        input_losses: jnp.array,
        hidden_column: jnp.array,
        receptive_area: int,
    ) -> jnp.array:
        """
        Args:
            input_losses: array of shape (input_dim)
            hidden_column: array of shape (hidden_dim * receptive_area,)

        Returns:
            array of shape (receptive_area, hidden_dim, input_dim)
        """
        input_losses = jnp.expand_dims(input_losses, axis=1)
        hidden_column = jnp.expand_dims(hidden_column, axis=1)
        # Shape = (input_col_dim, receptive_area * hidden_dim)
        # E.g., the shape of transposed parameters used in .backwards()
        updated_params = jnp.kron(input_losses, jnp.transpose(hidden_column))
        # Shape = (hidden_dim, hidden_dim, receptive_area * input_col_dim)
        # E.g., pre-transposed shape of the parameters
        reshaped_params = rearrange(
            updated_params,
            "i (r h) -> h (r i)",
            r=receptive_area,
        )
        return reshaped_params

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
            self.parameters = self.learn(
                input_activations,
                parameters,
                upward_mapping,
                downward_mapping,
                num_iters,
                learning_rate,
            )
        return self.forward(
            input_activiations=stride_inputs(input_activations, upward_mapping),
            parameters=self.parameters,
            input_is_one_hot=True,
            output_is_one_hot=True,
        )

    def __call__(
        self,
        *,
        input_activations: jnp.array,
        learn: bool,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
    ):
        if len(input_activations.shape) > 1:
            # vmap/pmap step over batch dimension
            pass

        return self.step(
            input_activations=input_activations,
            parameters=self.parameters,
            num_iters=self.num_iters,
            upward_mapping=upward_mapping,
            downward_mapping=downward_mapping,
            learning_rate=self.lr,
            learn=learn,
        )
