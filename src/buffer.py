from jax import jit, vmap
import jax.numpy as jnp

# Scaffolding for layerwise parallel memory buffer


class MemoryBuffer:
    def __init__(self, num_input_columns, config):
        self.enc_buffer = jnp.zeros(
            shape=(config.num_layers, num_input_columns, config.k_hot)
        )
        self.dec_buffer = jnp.zeros(
            shape=(config.num_layers, num_input_columns, config.hidden_column_dim)
        )


def nearest(buffer):
    return buffer[0]


def all(buffer):
    return buffer


def add_left(buffer, vector):
    return buffer.at[:, :, 1].set(vector)


def add_right(buffer, vector):
    return buffer.at[:, :, -1].set(vector)


def get_active_layers():
    pass


def get_feedback(buffer):
    return nearest(buffer)


def get_encoder_output(buffer):
    return nearest(buffer)


def get_layer_inputs(buffer):
    return nearest(buffer)


def get_prev_prediction(buffer):
    return nearest(buffer)
