import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dlwpt.trainer import Trainer
from dlwpt.utils import set_device, get_mnist_datasets


class CNN(nn.Module):
    def __init__(self, n_classes=10, n_filters=16, kernel_size=3, pool_size=2, optim=None):
        super().__init__()
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.loss_func = nn.CrossEntropyLoss()

        self.c = 1
        self.h = 28
        self.w = 28

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

    def predict(self, x):
        logits = self(x)
        return F.softmax(logits, dim=1)


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

    mod = CNN()
    opt = torch.optim.AdamW(mod.parameters(), lr=0.001)
    trainer = Trainer(
        mod, epochs=5, device=device, log_dir=LOG_DIR, checkpoint_file=LOG_DIR.joinpath('model.pt'),
        optimizer=opt
    )
    trainer.fit(train_loader, test_loader)
