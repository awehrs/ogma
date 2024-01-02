from src.network import Network

from pathlib import Path
import time

import jax.numpy as jnp
from omegaconf import OmegaConf
from tqdm import tqdm

config = OmegaConf.load(Path("src", "config") / "text.yaml")

# Build network.
network = Network.init_random(config)

print("Number of network parameters:", network.num_params)

# Learning.


# Create data.
def sin_wave(
    idx: int,
    window_len: int,
    delta_x: float,
    col_dim: int,
    upper_bound: int = 1,
    lower_bound: int = -1,
) -> jnp.array:
    x = jnp.linspace(start=idx, stop=idx + (window_len * delta_x), num=window_len)
    y = jnp.sin(x)
    y_binned = (
        (y - lower_bound) / (upper_bound - lower_bound) * (col_dim - 1) + 0.5
    ).astype(jnp.int16)

    if len(y_binned.shape) == 1:
        y_binned = jnp.expand_dims(y_binned, axis=-1)
    return y_binned


total_time = 0

DELTA_X = 0.02

for t in tqdm(range(1000)):
    value_to_encode = sin_wave(
        idx=0 + t * DELTA_X,
        window_len=config.x_dim,
        delta_x=DELTA_X,
        col_dim=config.preprocessor_dim,
    )
    t1 = time.time()
    network.step(precepts=value_to_encode, learn=True)
    t2 = time.time()
    total_time += t2 - t1

print("Average forward pass time:", total_time / 1000)

# Inference.
total_time = 0

for t in tqdm(range(500)):
    t1 = time.time()
    network.step(network.get_prediction(), learn=False)
    t2 = time.time()

    total_time += t2 - t1

    predicted_index = network.get_prediction()

    # print("Predicted index:", predicted_index[0])

print("Average inference pass took:", total_time / 500)

# key = random.PRNGKey(config.rng_seed)

# x_t = random.randint(
#     key, shape=(config.x_dim,), minval=0, maxval=config.preprocessor_dim
# )

# t1 = time.time()
# network.step(x_t)
# t2 = time.time()
# print(t2 - t1)
