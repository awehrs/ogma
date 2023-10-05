from functools import partial
from typing import Callable

from einops import rearrange
import jax
import jax.numpy as jnp
from jax import jit, vmap


@partial(jit, static_argnums=(1,))
def moving_window(x: jnp.array, size: int):
    """Because jnp doesn't have/need stride tricks. https://github.com/google/jax/issues/11354."""
    starts = jnp.arange(len(x) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(x, (start,), (size,)))(starts)


def stride_inputs(inputs: jnp.array, input_mapping: jnp.array) -> jnp.array:
    """
    Args:
        inputs: array of shape ([temporal_horizon,], n_input_columns, [input_column_dim]),
        depending on whether input is sparse or not
    Returns array of size (n_hidden_cols, receptive_area, [input_column_dim]),
        depending on whether input is sparse.
    """
    if inputs.shape == 3:
        # Multiple time steps of inputs.
        inputs = vmap(lambda x_t: x_t[input_mapping])(inputs)
        return rearrange(inputs, "t n r -> n (t r)")
    else:
        return inputs[input_mapping]


def compressed_to_full(idx: jnp.array, dim: int):
    """Construct one hot vector from the activation index."""
    matrix = jnp.zeros(shape=(len(idx), dim), dtype="int16")
    return matrix.at[jnp.arange(len(idx)), idx].set(1)


def concat_and_stride(
    arr_1: jnp.array, arr_2: jnp.array, mapping: jnp.array
) -> jnp.array:
    """Index align and stride to input arrays"""
    pass


def get_clock_schedule(clock_type: str) -> Callable:
    if clock_type == "exponential":

        def schedule(layer: int):
            return 2**layer

        return schedule

    else:
        raise NotImplementedError
