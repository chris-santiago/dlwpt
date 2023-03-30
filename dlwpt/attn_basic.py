from functools import partial

import numpy as np
import torch
import pytorch_lightning as L
import torchmetrics
import torch.nn as nn

from dlwpt.utils import get_mnist_datasets
from torch.utils.data import DataLoader, Dataset
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks.progress import RichProgressBar

from dlwpt.attn_baseline import BaselineNet, LargestDigit


class Flatten2(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1), -1)


class Combiner(nn.Module):
    def __init__(self, feature_net, weight_net):
        super().__init__()
        self.feature_net = feature_net
        self.weight_net = weight_net

    def forward(self, x):
        features = self.feature_net(x)
        weights = self.weight_net(features)

        if weights.dim() == 2:
            weights = weights.unsqueeze(2)
        weighted_features = features * weights
        return torch.sum(weighted_features, dim=1)


class BasicAttnNet(BaselineNet):
    def __init__(self, neurons=256, n_classes=10, sample_size=3, n_features=784, optim=None, loss_func=None, score_func=None):
        super().__init__()
        self.neurons = neurons
        self.n_classes = n_classes
        self.sample_size = sample_size
        self.n_features = n_features
        self.optim = optim
        self.loss_func = loss_func if loss_func else nn.CrossEntropyLoss()
        self.score_func = (
            score_func
            if score_func
            else torchmetrics.Accuracy(task="multiclass", num_classes=10)
        )

        self.save_hyperparameters()

        self.backbone = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(self.n_features, self.neurons),
            nn.LeakyReLU(),
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
            nn.Linear(self.neurons, 1),
            nn.Softmax(dim=1)
        )

        self.model = nn.Sequential(
            Combiner(self.backbone, self.attention),
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.n_classes),
        )


if __name__ == '__main__':
    BATCH = 128
    NUM_WORKERS = 10
    EPOCHS = 10

    train, test = get_mnist_datasets()
    train = LargestDigit(train)
    test = LargestDigit(test)

    train_dl = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test, batch_size=BATCH, num_workers=NUM_WORKERS)

    optim = partial(torch.optim.Adam, lr=0.001)
    mod = BasicAttnNet(optim=optim)

    # track experimental data by using Aim
    aim_logger = AimLogger(
        experiment="basic-attn-largest-digit",
        train_metric_prefix="train_",
        val_metric_prefix="valid_",
    )

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="mps",
        devices=1,
        logger=aim_logger,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )
    trainer.fit(mod, train_dataloaders=train_dl, val_dataloaders=test_dl)
