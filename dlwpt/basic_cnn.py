import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dlwpt.utils import set_device, get_mnist_datasets

BATCH = 32


class CNN(nn.Module):
    def __init__(self, lr=0.01, n_classes=10, n_filters=16, kernel_size=3, optim=None):
        super().__init__()
        self.lr = lr
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.loss_func = nn.CrossEntropyLoss()

        self.c = 1
        self.h = 28
        self.w = 28

        self.layers = nn.Sequential(
            nn.Conv2d(self.c, self.n_filters, self.kernel_size, padding=self.kernel_size // 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.n_filters * self.h * self.w, self.n_classes)
        )

        self.optim = optim if optim else torch.optim.Adam(self.layers.parameters(), lr=self.lr)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, idx):
        x, y = batch
        pred = self.forward(x)
        loss = self.loss_func(pred, y)
        return loss


class Trainer:
    def __init__(self, model, epochs=20, device=None):
        self.model = model
        self.epochs = epochs
        self.device = device if device else set_device()
        self.model.to(self.device)
        self.total_loss = 0

    def fit(self, train_dl):
        for epoch in tqdm.tqdm(range(self.epochs), desc='Epoch'):
            self.model.train()

            for inputs, labels in train_dl:
                inputs.to(self.device)
                labels.to(self.device)
                self.model.optim.zero_grad()

                pred = self.model(inputs)
                loss = self.model.loss_func(pred, labels)
                loss.backward()
                self.model.optim.step()
                self.total_loss += loss.item()


if __name__ == "__main__":
    train, test = get_mnist_datasets()

    train_loader = DataLoader(train, batch_size=BATCH, shuffle=True)
    test_loader = DataLoader(test, batch_size=BATCH)

    mod = CNN()
    trainer = Trainer(mod, epochs=5)
    trainer.fit(train_loader)
    print(trainer.total_loss)


