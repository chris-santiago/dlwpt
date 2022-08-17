import torch
import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy

from dlwpt.utils import set_device


class Trainer:
    def __init__(self, model, epochs=20, device=None, log_dir=None):
        self.model = model
        self.epochs = epochs
        self.device = device if device else set_device()
        self.model.to(self.device)
        self.metric = Accuracy()
        self.writer = SummaryWriter(log_dir=log_dir)

    def fit(self, train_dl, test_dl=None):
        for epoch in tqdm.tqdm(range(self.epochs), desc='Epoch', leave=True):
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

                preds = self.model.predict(inputs)
                self.metric(preds.cpu(), labels.cpu())

            train_acc = self.metric.compute()
            self.metric.reset()
            self.writer.add_scalar('TrainLoss', total_loss, epoch)
            self.writer.add_scalar('TrainAccuracy', train_acc, epoch)

            if test_dl:
                self.model.eval()
                with torch.no_grad():
                    total_loss = 0

                    for inputs, labels in tqdm.tqdm(test_dl, desc='Batch', leave=False):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        logits = self.model(inputs)
                        loss = self.model.loss_func(logits, labels)
                        total_loss += loss.item()

                        preds = self.model.predict(inputs)
                        self.metric(preds.cpu(), labels.cpu())

                    valid_acc = self.metric.compute()
                    self.metric.reset()
                    self.writer.add_scalar('ValidLoss', total_loss, epoch)
                    self.writer.add_scalar('ValidAccuracy', valid_acc, epoch)

        self.writer.flush()
        self.writer.close()
