from src.encoder import Encoder
from src.decoder import Decoder
from src.network import Network
from src.utils import stride_inputs

from pathlib import Path

import jax
import jax.numpy as jnp
from jax import random
from omegaconf import OmegaConf

config = OmegaConf.load(Path("src", "config") / "text.yaml")

network = Network.init_random(config)

layer = network.layers[0]
upward_mapping = network.upward_mapping
downward_mapping = network.downward_mapping

encoder = layer.enc
decoder = layer.dec

key = random.PRNGKey(config.rng_seed)

x_t = random.randint(
    key, shape=(config.x_dim,), minval=0, maxval=config.preprocessor_dim
)

network.step(x_t)
