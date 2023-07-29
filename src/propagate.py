from functools import partial

from einops import rearrange
import jax.numpy as jnp
from jax import jit, vmap


def propagate(
    inputs: jnp.array,
    parameters: jnp.array,
    input_is_one_hot: bool,
    output_is_one_hot: bool,
):
    """
    Backward and forward propagation. Either dense or sparse batch matrix multiply.

    Args:
        inputs:
            array of shape (num_hidden_columns, receptive_area) [if input is one hot],
            or of shape (num_hidden_columns, receptive_area, input_dim) [input isn't one hot]
        parameters: array of shape (num_output_columns, output_dim, receptive_area * input_dim)
        output_is_one_hot: whether or not to one_hot the output

    Returns:
        array of shape (num_output_columns,) if output_is_one_hot,
            else, (num_output_columns, output_dim)
    """
    if input_is_one_hot:
        output = vmap(sparse_matmul)(parameters, inputs)
    else:
        inputs = rearrange(inputs, "n r d -> n (r d)")
        output = vmap(lambda W, x: jnp.dot(W, x))(parameters, inputs)
    if output_is_one_hot:
        output = jnp.argmax(output, axis=1)

    return output


@partial(jit, static_argnums=(1,))
def sparse_matmul(
    parameters: jnp.array,
    input_activations: jnp.array,
):
    """
    Args:
        input_activations: array of shape (receptive_area,)
            Values indicate the row number of the active cell in each input column.
        parameters: array of shape,
            (output_column_dim, receptive_area * input_column_dim)

    Returns:
        Index of maximum activation in output column. Array of size (output_column_dim,).
    """
    receptive_area = len(input_activations)
    input_dim = parameters.shape[-1] // receptive_area

    offset = jnp.arange(start=0, stop=(receptive_area * input_dim), step=input_dim)
    idx = input_activations + offset

    unsummed = jnp.take(parameters, idx, axis=-1)

    return jnp.sum(unsummed, axis=1)
