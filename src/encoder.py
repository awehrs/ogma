from src.utils import dense_to_sparse, sparse_to_dense, stride_inputs
from src.propagate import propagate

from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Optional

from einops import rearrange
from jax import vmap, jit
import jax.numpy as jnp


class Encoder(ABC):
    """Abstract base class for stateless encoder."""

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass

    @abstractmethod
    def learn():
        pass

    @abstractmethod
    def update():
        pass


class ReconstructionEncoder(Encoder):
    """Exponential Reconstruction Encoder."""

    def forward(
        input_activations: jnp.array,
        parameters: jnp.array,
        k_hot_input: int,
        k_hot_output: int,
    ) -> jnp.array:
        """
        Args:
            input_activations: array of shape = (num_columns, k_hot_input)
            parameters: array of shape =
                (num_columns, hidden_column_dim, receptive_area * input_dim)
            k_hot_input: number of active indices per input column.
            k_hot_output: number of active indices per output column.
        Returns:
            array of shape (num_columns, k_hot_output)
        """
        return propagate(
            input_activations,
            parameters,
            k_hot_input=k_hot_input,
            k_hot_output=k_hot_output,
        )

    def backward(
        output_activations: jnp.array,
        parameters: jnp.array,
        downward_mapping: jnp.array,
        k_hot_input: int,
        k_hot_output: int,
    ) -> jnp.array:
        """
        Args:
            output_activations: array of shape (num_columns, k_hot_input)
            parameters: array of shape =
                (num_columns, hidden_column_dim, receptive_area * input_dim)
            downward_mapping: array of shape (num_columns, receptive_area, k_hot_input)
            k_hot_input: number of active indices per input column.
            k_hot_output: number of active indices per output column.
        Returns:
            array of shape (num_columns, input_dim)
        """
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
        input_activations: jnp.array,
        parameters: jnp.array,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
        num_iters: int,
        k_hot: int,
        learning_rate: float,
        input_column_dim: int,
    ) -> Tuple[jnp.array]:
        """
        Args:
            input_activations: array of shape (num_columns, receptive_area, k_hot)
        Returns:
            parameters: array of shape (num_columns, output_dim, receptive_area * input_dim)
            reconstruction: array of shape (num_columns, input_column_dim)
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

            output += ReconstructionEncoder.forward(
                inputs,
                parameters,
                k_hot_input=k_hot if i < 1 else None,
                k_hot_output=k_hot,
            )

            activation = dense_to_sparse(output, k_hot)

            recons = ReconstructionEncoder.backward(
                activation,
                parameters,
                downward_mapping,
                k_hot_input=k_hot,
                k_hot_output=None,
            )

        loss = ReconstructionEncoder.loss(input_activations, recons)

        hidden = jnp.argmax(activation, axis=1)

        hidden = sparse_to_dense(hidden, dim=hidden_col_dim, k_hot=k_hot)

        hidden = stride_inputs(hidden, downward_mapping)

        hidden = rearrange(hidden, "n r h -> n (r h)")

        update = vmap(ReconstructionEncoder.update, in_axes=(0, 0, None))

        delta = update(
            loss,
            hidden,
            upward_mapping.shape[-1],
        )

        parameters += learning_rate * delta

        return parameters, recons

    def loss(inputs: jnp.array, reconstruction: jnp.array) -> jnp.array:
        """
        Args:
            inputs: array of shape (num_columns, input_dim)
            reconstruction: array of shape (num_columns, input_dim)
        """
        return inputs - reconstruction

    def update(
        input_losses: jnp.array,
        hidden_column: jnp.array,
        receptive_area: int,
    ) -> jnp.array:
        """
        Args:
            input_losses: array of shape (input_dim)
            hidden_column: array of shape (hidden_dim * receptive_area,)
            receptive_area: number of input columns to which output column is connected.
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

    @staticmethod
    @partial(
        jit,
        static_argnames=[
            "k_hot",
            "learn",
            "input_column_dim",
            "learning_rate",
            "num_iters",
        ],
    )
    def step(
        input_activations: jnp.array,
        parameters: jnp.array,
        upward_mapping: jnp.array,
        downward_mapping: jnp.array,
        k_hot: int,
        learn: bool,
        input_column_dim: Optional[int] = None,
        learning_rate: Optional[float] = None,
        num_iters: Optional[int] = None,
    ) -> Tuple[Optional[jnp.array]]:
        """
        Args:
            input_activations: array of shape (num_columns, k_hot)
            parameters: array of shape (num_columns, output_dim, receptive_area * output_dim)
            upward_mapping: array of shape (num_columns, receptive_area, k_hot)
            downward_mapping: array of shape (num_columns, receptive_area, k_hot)
            k_hot: number of active cells per column.
            learn: whether to perform reconstruction learning.
            input_column_dim: dense dimension of inputs.
            learning_rate: factor by which to update weights.
            num_iters: number of reconstruction loops to perform.
        Returns:
            parameters: updated parameters, or NoneType if learn == False.
            output: array of shape (num_columns, k_hot).
            reconstruction: array of shape (num_columns, input_column_dim).
        """
        if learn:
            parameters, reconstruction = ReconstructionEncoder.learn(
                input_activations=input_activations,
                parameters=parameters,
                upward_mapping=upward_mapping,
                downward_mapping=downward_mapping,
                num_iters=num_iters,
                k_hot=k_hot,
                learning_rate=learning_rate,
                input_column_dim=input_column_dim,
            )
        else:
            reconstruction = None

        output = ReconstructionEncoder.forward(
            input_activations=stride_inputs(input_activations, upward_mapping),
            parameters=parameters,
            k_hot_input=k_hot,
            k_hot_output=k_hot,
        )

        return parameters, output, reconstruction
