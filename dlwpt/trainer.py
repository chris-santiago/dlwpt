from collections import defaultdict

import torch
import tqdm
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy

from dlwpt.utils import set_device


class Trainer:
    def __init__(
            self, model, epochs=20, score_funcs=None, device=None, log_dir=None,
            checkpoint_file=None, optimizer=None, lr_schedule=None
    ):
        self.model = model
        self.epochs = epochs
        self.score_funcs = score_funcs if score_funcs else {'accuracy': Accuracy()}
        self.device = device if device else set_device()
        self.model.to(self.device)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.checkpoint_file = checkpoint_file
        self.optimizer = optimizer if optimizer else torch.optim.AdamW(self.model.parameters())
        self.lr_schedule = lr_schedule
        self.results = defaultdict(list)

    def score_batch(self, inputs, labels):
        preds = self.model.predict(inputs)
        for metric in self.score_funcs:
            self.score_funcs[metric](preds.cpu(), labels.cpu())

    def score_epoch(self, total_loss, epoch, mode):
        self.results[f'{mode}_loss'].append(total_loss)
        self.writer.add_scalar(f'{mode}_loss', total_loss, epoch)
        for metric in self.score_funcs:
            score = self.score_funcs[metric].compute().item()
            self.results[f'{mode}_{metric}'].append(score)
            self.writer.add_scalar(f'{mode}_{metric}', score, epoch)
            self.score_funcs[metric].reset()

    def fit(self, train_dl, valid_dl=None):
        for epoch in tqdm.tqdm(range(self.epochs), desc='Epoch', leave=True):
            self.results['epoch'].append(epoch)
            self.model.train()
            total_loss = 0

            for inputs, labels in tqdm.tqdm(train_dl, desc='Batch', leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()

                logits = self.model(inputs)
                loss = self.model.loss_func(logits, labels)
                loss.backward()
                total_loss += loss.item()
                self.optimizer.step()
                self.score_batch(inputs, labels)

            total_loss /= len(train_dl)
            self.score_epoch(total_loss, epoch, 'train')

            if valid_dl:
                self.model.eval()
                with torch.no_grad():
                    total_loss = 0
                    for inputs, labels in tqdm.tqdm(valid_dl, desc='Batch', leave=False):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        logits = self.model(inputs)
                        loss = self.model.loss_func(logits, labels)
                        total_loss += loss.item()
                        self.score_batch(inputs, labels)

                    total_loss /= len(valid_dl)
                    self.score_epoch(total_loss, epoch, 'valid')

                if self.lr_schedule:
                    if isinstance(self.lr_schedule, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_schedule.step(total_loss)
                    else:
                        self.lr_schedule.step()

            if self.checkpoint_file is not None:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'results': self.results
                    },
                    self.checkpoint_file
                )

        self.writer.flush()
        self.writer.close()
        return self
