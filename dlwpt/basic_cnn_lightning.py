"""
Basic CNN refactored with PyTorch Lightning.

The original version included a scratch trainer for fun / learning and following along with book.
This version implements PTL 2.0 and Aim (experiment tracking), similar to use in Emonet model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader
import pytorch_lightning as L
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks.progress import RichProgressBar

from dlwpt.utils import set_device, get_mnist_datasets


class LitCNN(L.LightningModule):
    def __init__(self, n_classes=10, n_filters=16, kernel_size=3, pool_size=2):
        super().__init__()
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.loss_func = nn.CrossEntropyLoss()
        self.score_func = torchmetrics.Accuracy(task='multiclass', num_classes=10)

        self.c = 1
        self.h = 28
        self.w = 28

        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Conv2d(self.c, self.n_filters, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_filters, self.n_filters, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),
            nn.Conv2d(self.n_filters, self.n_filters*self.pool_size, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_filters*self.pool_size, self.n_filters*self.pool_size, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.Conv2d(self.n_filters*self.pool_size, self.n_filters*self.pool_size, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.MaxPool2d(self.pool_size),
            nn.Flatten(),
            nn.Linear((self.n_filters * self.h * self.w) // ((self.pool_size*self.pool_size)*2), self.n_classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_func(preds, y)
        score = self.score_func(preds, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_func(preds, y)
        score = self.score_func(preds, y)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_accuracy', score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


if __name__ == "__main__":
    from dlwpt import ROOT
    from datetime import datetime

    NOW = datetime.now().strftime('%Y%m%d-%H%M')
    LOG_DIR = ROOT.joinpath('runs', NOW)
    BATCH = 128
    train, test = get_mnist_datasets(do_augment=False)

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH)
    device = set_device()

    # track experimental data by using Aim
    aim_logger = AimLogger(
        experiment='aim_on_pt_lightning',
        train_metric_prefix='train_',
        val_metric_prefix='valid_',
    )

    mod = LitCNN()
    trainer = L.Trainer(max_epochs=5, accelerator='mps', devices=-1, logger=aim_logger,
                        callbacks=[RichProgressBar(refresh_rate=5, leave=True)]
                        )
    trainer.fit(mod, train_dataloaders=train_loader, val_dataloaders=test_loader)
