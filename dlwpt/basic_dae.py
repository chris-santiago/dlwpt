from typing import Tuple

import torch.nn as nn

from dlwpt.basic_autoencoder import AutoEncoder
from dlwpt.utils import WhiteNoise


class DenoisingAutoEncoder(AutoEncoder):
    def __init__(self, layers: Tuple[int, ...], input_shape: Tuple[int, int]):
        super().__init__(layers, input_shape)
        
        self.encoder = nn.Sequential(
            nn.Flatten(),
            WhiteNoise(),
            nn.Linear(self.input_shape[0] * self.input_shape[1], self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1])
        
        )
