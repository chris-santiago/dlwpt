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


class LargestDigit(Dataset):
    def __init__(self, dataset, sample_size=3):
        self.dataset = dataset
        self.sample_size = sample_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        indices = np.random.randint(0, len(self.dataset), size=self.sample_size)
        x = torch.stack([self.dataset[i][0] for i in indices])
        y = max([self.dataset[i][1] for i in indices])
        return x, y


class BaselineNet(L.LightningModule):
    def __init__(self, neurons=256, n_classes=10, optim=None, loss_func=None, score_func=None):
        super().__init__()
        self.neurons = neurons
        self.n_classes = n_classes
        self.optim = optim
        self.loss_func = loss_func if loss_func else nn.CrossEntropyLoss()
        self.score_func = (
            score_func
            if score_func
            else torchmetrics.Accuracy(task="multiclass", num_classes=10)
        )

        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784*3, self.neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.n_classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_func(preds, y)
        score = self.score_func(preds, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_accuracy",
            score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_func(preds, y)
        score = self.score_func(preds, y)
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "valid_accuracy",
            score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # return loss

    def configure_optimizers(self):
        return self.optim(self.parameters())


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
    mod = BaselineNet(optim=optim)

    # track experimental data by using Aim
    aim_logger = AimLogger(
        experiment="baseline-largest-digit",
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
