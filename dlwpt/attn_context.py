from functools import partial

import numpy as np
import torch
import pytorch_lightning as L
import torchmetrics
import aim
import torch.nn as nn
import torch.nn.functional as F

from dlwpt.utils import get_mnist_datasets
from torch.utils.data import DataLoader
from aim.pytorch_lightning import AimLogger
from pytorch_lightning.callbacks.progress import RichProgressBar
from dlwpt.attn_basic import LargestDigit, BaselineNet


class DotScore(nn.Module):
    def __init__(self, h_dims):
        super().__init__()
        self.h_dims = h_dims

    def forward(self, states, context):
        scores = torch.bmm(states, context.unsqueeze(2)) / np.sqrt(self.h_dims)
        return scores


class ApplyAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states, attn_scores, mask=None):
        if mask is not None:
            attn_scores[mask] = -1000
        weights = F.softmax(attn_scores, dim=1)
        final_context = (states*weights).sum(dim=1)
        return final_context, weights


def get_mask(x, time_dim=1, fill=0):
    sum_over = list(range(1, len(x.shape)))
    if time_dim in sum_over:
        sum_over.remove(time_dim)
    with torch.no_grad():
        mask = torch.sum((x != fill), dim=sum_over) > 0
    return mask


class AttnContextNet(BaselineNet):
    def __init__(self, neurons=256, n_classes=10, sample_size=3, n_features=784, optim=None,
                 loss_func=None, score_func=None):
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

        self.attention = ApplyAttention()

        self.score_net = DotScore(self.neurons)

        self.model = nn.Sequential(
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.neurons),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.neurons),
            nn.Linear(self.neurons, self.n_classes),
        )

    def forward(self, x):
        mask = get_mask(x)
        features = self.backbone(x)
        context = (mask.unsqueeze(-1) * features).sum(dim=1)
        context = context / (mask.sum(dim=1).unsqueeze(-1) + 1e-10)
        scores = self.score_net(features, context)
        final_context, _ = self.attention(features, scores, mask=mask)
        return self.model(final_context)

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
        log_img = torch.rand(1)
        if log_img > .95:  # sample 5% of time  # todo make this a function of the score
            label = str(y[0].item())
            for i in range(3):
                img = aim.Image(x[0, i, :, :, :], caption=label)
                self.logger.experiment.track(
                    img,
                    name=f'Image Set, Sample {str(log_img.item())}',
                    context={'dataset': 'train', 'max': label}
                )


if __name__ == '__main__':
    BATCH = 128
    NUM_WORKERS = 10
    EPOCHS = 10
    PLOT_MODEL = False

    train, test = get_mnist_datasets()
    train = LargestDigit(train)
    test = LargestDigit(test)

    train_dl = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS)
    test_dl = DataLoader(test, batch_size=BATCH, num_workers=NUM_WORKERS)

    optim = partial(torch.optim.Adam, lr=0.001)
    mod = AttnContextNet(optim=optim)

    if PLOT_MODEL:
        from torchview import draw_graph

        graph = draw_graph(
            mod, input_size=(BATCH, 3, 1, 28, 28), device='mps', expand_nested=False,
            save_graph=True, depth=5, directory='../images/'
        )
        graph.visual_graph.render(format='svg')

    # track experimental data by using Aim
    aim_logger = AimLogger(
        experiment="attn-context-largest-digit",
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
