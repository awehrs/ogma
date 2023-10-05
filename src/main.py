from src.encoder import Encoder
from src.decoder import Decoder
from src.network import Network

from pathlib import Path
import time

import jax.numpy as jnp
from jax import random
from omegaconf import OmegaConf

config = OmegaConf.load(Path("src", "config") / "text.yaml")

network = Network.init_random(config)

layer = network.layers[0]
upward_mapping = network.upward_mapping
downward_mapping = network.downward_mapping

encoder = layer.encoder
decoder = layer.decoder

key = random.PRNGKey(config.rng_seed)

x_t = random.randint(
    key, shape=(config.x_dim,), minval=0, maxval=config.preprocessor_dim
)

print(network.num_params)

t1 = time.time()
network.step(x_t)
t2 = time.time()
print(t2 - t1)
