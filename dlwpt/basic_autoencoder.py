from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError

from dlwpt.trainer import Trainer
from dlwpt.utils import set_device, get_mnist_datasets, WhiteNoise


class EncoderLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.size_in = input_size
        self.size_out = output_size
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)


class AutoEncoder(nn.Module):
    def __init__(self, layers: Tuple[int, ...], input_shape: Tuple[int, int]):
        super().__init__()
        self.input_shape = input_shape
        self.encoder_shape = layers
        self.decoder_shape = tuple(reversed(layers))
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(self.encoder_shape[i], self.encoder_shape[i + 1])
                for i in range(len(self.encoder_shape) - 2)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                EncoderLayer(self.decoder_shape[i], self.decoder_shape[i + 1])
                for i in range(len(self.decoder_shape) - 2)
            ]
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_shape[0] * self.input_shape[1], self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1])
        )
        self.decoder = nn.Sequential(
            *self.decoder_layers,
            nn.Linear(self.decoder_shape[-2], self.decoder_shape[-1]),
            nn.Linear(self.decoder_shape[-1], self.input_shape[0] * self.input_shape[1])
        )
        self.loss_func = nn.MSELoss()

    def forward(self, x):
        input_shape = x.shape
        out = self.decoder(self.encoder(x))
        return out.reshape(-1, *input_shape[1:])

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)


class NoisyAutoEncoder(AutoEncoder):
    def __init__(self, layers: Tuple[int, ...], input_shape: Tuple[int, int]):
        super().__init__(layers, input_shape)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            WhiteNoise(),
            nn.Linear(self.input_shape[0] * self.input_shape[1], self.encoder_shape[0]),
            *self.encoder_layers,
            nn.Linear(self.encoder_shape[-2], self.encoder_shape[-1])
        )


class AutoEncoderDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inputs, _ = self.dataset.__getitem__(idx)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs, inputs


if __name__ == "__main__":
    from dlwpt import ROOT
    from datetime import datetime

    NOW = datetime.now().strftime("%Y%m%d-%H%M")
    LOG_DIR = ROOT.joinpath("runs", NOW)
    BATCH = 128
    train, test = get_mnist_datasets(do_augment=False)
    train, test = AutoEncoderDataset(train), AutoEncoderDataset(test)

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH)
    device = set_device()

    mod = AutoEncoder(layers=(128, 64, 16), input_shape=(28, 28))
    opt = torch.optim.AdamW(mod.parameters(), lr=0.001)
    trainer = Trainer(
        mod,
        epochs=10,
        device=device,
        log_dir=LOG_DIR,
        checkpoint_file=LOG_DIR.joinpath("model.pt"),
        optimizer=opt,
        score_funcs={"mse": MeanSquaredError()},
    )
    trainer.fit(train_loader, test_loader)
