from src.utils import dense_to_sparse

from functools import partial
from typing import Optional

from einops import rearrange
import jax.numpy as jnp
from jax import jit, vmap


def propagate(
    inputs: jnp.array,
    parameters: jnp.array,
    k_hot_input: Optional[int] = None,
    k_hot_output: Optional[int] = None,
) -> jnp.array:
    """
    Backward and forward propagation. Either dense or sparse batch matrix multiply.

    Args:
        inputs:
            array of shape (num_hidden_columns, receptive_area) [if input is one hot],
            or of shape (num_hidden_columns, receptive_area, input_dim) [input isn't one hot]
        parameters: array of shape (num_output_columns, output_dim, receptive_area * input_dim)

    Returns:
        array of shape (num_output_columns,) if output_is_one_hot,
            else, (num_output_columns, output_dim)
    """
    if k_hot_input is not None:
        output = vmap(sparse_matmul, in_axes=(0, 0, None))(
            parameters, inputs, k_hot_input
        )
    else:
        inputs = rearrange(inputs, "n r d -> n (r d)")
        output = vmap(lambda W, x: jnp.dot(W, x))(parameters, inputs)

    if k_hot_output is not None:
        output = dense_to_sparse(output, k_hot_output)

    return output


@partial(jit, static_argnames=["k_hot"])
def sparse_matmul(
    parameters: jnp.array,
    input_activations: jnp.array,
    k_hot: int,
):
    """
    Args:
        input_activations: array of shape (receptive_area, k_hot)
            Values indicate the row numbers of the active cell in each input column.
        parameters: array of shape,
            (output_column_dim, receptive_area * input_column_dim)
        k_hot:

    Returns:
        Index of maximum activation in output column. Array of size (output_column_dim,).
    """
    receptive_area = len(input_activations)
    input_dim = parameters.shape[-1] // receptive_area
    offset = jnp.arange(start=0, stop=(receptive_area * input_dim), step=input_dim)
    offset = jnp.repeat(offset, repeats=k_hot, axis=0)

    if k_hot == 1:
        offset = jnp.expand_dims(offset, axis=-1)

    idx = input_activations + offset
    idx = rearrange(idx, "r k -> (r k)")

    unsummed = jnp.take(parameters, idx, axis=-1)

    return jnp.sum(unsummed, axis=1)
