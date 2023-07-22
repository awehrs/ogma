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
upward_mapping = layer.upward_mapping
downward_mapping = layer.downward_mapping

encoder = layer.enc

key = random.PRNGKey(config.rng_seed)

x_t = random.randint(
    key, shape=(config.x_dim,), minval=0, maxval=config.preprocessor_dim
)

for l in network.layers:
    y_t = l.enc.step(
        x_t,
        parameters=encoder.parameters,
        upward_mapping=upward_mapping,
        downward_mapping=downward_mapping,
        num_iters=config.num_iters,
        learning_rate=config.encoder_lr,
        learn=True,
    )

    print(y_t - x_t)
    x_t = y_t
