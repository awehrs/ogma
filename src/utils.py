from functools import partial


import jax
import jax.numpy as jnp
from jax import jit, lax, vmap


@partial(jit, static_argnums=(1,))
def moving_window(x: jnp.array, size: int):
    """Because jnp doesn't have/need stride tricks. https://github.com/google/jax/issues/11354."""
    starts = jnp.arange(len(x) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(x, (start,), (size,)))(starts)


def stride_inputs(
    inputs: jnp.array,
    input_mapping: jnp.array,
) -> jnp.array:
    """
    Args:
        - inputs: array of shape (n_input_columns, [input_column_dim/k-hot]),
        depending on whether input is one-hot, dense, or k-sparse.
        - input_mapping: array of shape (num_hidden_columsn, receptive_area)
    Returns array of size (n_hidden_cols, receptive_area, [input_column_dim/k-hot]),
        depending on whether input is one-hot, dense, or k-sparse.
    """
    return inputs[input_mapping]


def sparse_to_dense(idx: jnp.array, dim: int, k_hot: int) -> jnp.array:
    """
    Construct dense vector from the activation index. Only called on
        activations that have had dimensions adjusted.

    Args:
        idx: array of shape (num_hidden_columns, k_hot)
    """
    matrix = jnp.zeros(shape=(len(idx), dim), dtype="int16")
    row_idx = jnp.expand_dims(jnp.arange(len(idx)), axis=1)

    return matrix.at[row_idx, idx].set(1, mode="drop")


def dense_to_sparse(arr: jnp.array, k_hot: int) -> jnp.array:
    """Activate layer of dense columns to be k-hot"""
    if k_hot == 1:
        return jnp.expand_dims(jnp.argmax(arr, axis=1), axis=-1).astype(jnp.int16)
    else:
        return lax.top_k(arr, k=k_hot)[1].astype(jnp.int16)
