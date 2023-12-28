from src.network import Network

import numpy as np


class Trainer:
    def __init__(self, config, network: Network, optimizer):
        self.config = config
        self.network = network
        self.optimizer = optimizer
        self.enviornment = None

        self.forward_fn = (
            network.naive_async_step if config.async_step else network.step
        )

    def train(self):
        for step in self.config.max_steps:
            self.network.forward_fn()

    def eval(self):
        pass

    def inner_traning_loop(self):
        pass

    def __call__(self):
        pass
