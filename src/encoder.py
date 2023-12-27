from src.utils import dense_to_sparse, sparse_to_dense, stride_inputs
from src.propagate import propagate

from typing import Tuple, Optional

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
        input_activations: jnp.array,
        parameters: jnp.array,
        k_hot_input: int,
        k_hot_output: int,
    ) -> jnp.array:
        return propagate(
            input_activations,
            parameters,
            k_hot_input=k_hot_input,
            k_hot_output=k_hot_output,
        )

    def backward(
        self,
        output_activations: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        k_hot_input: int,
        k_hot_output: int,
    ) -> jnp.array:
        output_activations = stride_inputs(output_activations, downward_mapping)
        parameters = rearrange(
            parameters, "n h (r i) -> n i (r h)", r=output_activations.shape[1]
        )
        recons = propagate(
            output_activations,
            parameters,
            k_hot_input=k_hot_input,
            k_hot_output=k_hot_output,
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
        k_hot: int,
        learning_rate: float,
        input_column_dim: int,
    ) -> jnp.array:
        """
        Args:
            input_activations: tensor of shape (num_hidden_columns, receptive_area, k_hot)
        """
        hidden_col_dim = parameters.shape[1]

        # (num_cols, hidden_dim)
        output = jnp.zeros(
            shape=(parameters.shape[1], parameters.shape[1]), dtype=jnp.int16
        )

        # (num_cols, input_dim)
        recons = jnp.zeros(
            shape=(
                parameters.shape[1],
                input_column_dim,
            )
        )

        for i in range(num_iters):
            if i == 0:
                inputs = stride_inputs(input_activations, upward_mapping)
                # Expand to allow binary op with recons in next iter
                input_activations = sparse_to_dense(
                    input_activations,
                    dim=input_column_dim,
                    k_hot=k_hot,
                )
            else:
                inputs = stride_inputs(input_activations - recons, upward_mapping)

            output += self.forward(
                inputs,
                parameters,
                k_hot_input=k_hot if i < 1 else None,
                k_hot_output=k_hot,
            )

            activation = dense_to_sparse(output, k_hot)

            recons = self.backward(
                activation,
                parameters,
                downward_mapping,
                k_hot_input=k_hot,
                k_hot_output=None,
            )

        loss = self.loss(input_activations, recons)

        hidden = jnp.argmax(activation, axis=1)

        hidden = sparse_to_dense(hidden, dim=hidden_col_dim, k_hot=k_hot)

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
        k_hot: int,
        learning_rate: float,
        input_column_dim: int,
        learn: bool = True,
    ) -> Tuple[jnp.array]:
        if learn:
            self.parameters = self.learn(
                input_activations=input_activations,
                parameters=parameters,
                upward_mapping=upward_mapping,
                downward_mapping=downward_mapping,
                num_iters=num_iters,
                k_hot=k_hot,
                learning_rate=learning_rate,
                input_column_dim=input_column_dim,
            )
        return self.forward(
            input_activations=stride_inputs(input_activations, upward_mapping),
            parameters=self.parameters,
            k_hot_input=k_hot,
            k_hot_output=k_hot,
        )

    def __call__(
        self,
        *,
        input_activations: jnp.array,
        k_hot: int,
        learn: bool,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
        input_column_dim: Optional[int] = None,
    ):
        if len(input_activations.shape) > 1:
            # vmap/pmap step over batch dimension
            pass

        return self.step(
            input_activations=input_activations,
            parameters=self.parameters,
            upward_mapping=upward_mapping,
            downward_mapping=downward_mapping,
            num_iters=self.num_iters,
            k_hot=k_hot,
            learning_rate=self.lr,
            input_column_dim=input_column_dim,
            learn=learn,
        )
