import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy

from dlwpt.utils import set_device, get_mnist_datasets


class CNN(nn.Module):
    def __init__(self, lr=0.01, n_classes=10, n_filters=16, kernel_size=3, pool_size=2, optim=None):
        super().__init__()
        self.lr = lr
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.loss_func = nn.CrossEntropyLoss()

        self.c = 1
        self.h = 28
        self.w = 28

        self.layers = nn.Sequential(
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

        self.optim = optim if optim else torch.optim.Adam(self.layers.parameters(), lr=self.lr)

    def forward(self, x):
        return self.layers(x)


class Trainer:
    def __init__(self, model, epochs=20, device=None, log_dir=None):
        self.model = model
        self.epochs = epochs
        self.device = device if device else set_device()
        self.model.to(self.device)
        self.metric = Accuracy()
        self.writer = SummaryWriter(log_dir=log_dir)

    def fit(self, train_dl, test_dl=None):
        for epoch in tqdm.tqdm(range(self.epochs), desc='Epoch'):
            self.model.train()
            total_loss = 0

            for inputs, labels in tqdm.tqdm(train_dl, desc='Batch', leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.model.optim.zero_grad()

                logits = self.model(inputs)
                loss = self.model.loss_func(logits, labels)
                loss.backward()
                total_loss += loss.item()
                self.model.optim.step()

                preds = F.softmax(logits.cpu(), dim=1).argmax(1)
                self.metric(preds, labels.cpu())

            train_acc = self.metric.compute()
            self.metric.reset()
            self.writer.add_scalar('TrainLoss', total_loss, epoch)
            self.writer.add_scalar('TrainAccuracy', train_acc, epoch)

            if test_dl:
                self.model.eval()
                with torch.no_grad():
                    total_loss = 0

                    for inputs, labels in tqdm.tqdm(train_dl, desc='Batch', leave=False):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        logits = self.model(inputs)
                        loss = self.model.loss_func(logits, labels)
                        total_loss += loss.item()

                        preds = F.softmax(logits.cpu(), dim=1).argmax(1)
                        self.metric(preds, labels.cpu())

                    valid_acc = self.metric.compute()
                    self.metric.reset()
                    self.writer.add_scalar('ValidLoss', total_loss, epoch)
                    self.writer.add_scalar('ValidAccuracy', valid_acc, epoch)

        self.writer.flush()
        self.writer.close()


if __name__ == "__main__":
    from dlwpt import ROOT
    from datetime import datetime

    NOW = datetime.now().strftime('%Y%m%d-%H%M')
    LOG_DIR = ROOT.joinpath('runs', NOW)
    BATCH = 128
    train, test = get_mnist_datasets()

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH)
    device = set_device()

    mod = CNN()
    trainer = Trainer(mod, epochs=5, device=device, log_dir=LOG_DIR)
    trainer.fit(train_loader, test_loader)


