from functools import partial

from pytorch_lightning.callbacks import RichProgressBar
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import pytorch_lightning as L
from aim.pytorch_lightning import AimLogger

from dlwpt.utils import get_shakespear_data


class AutoRegressiveDataset(Dataset):
    def __init__(self, doc: str, vocab: Vocab, max_chunk: int = 500):
        self.doc = doc
        self.vocab = vocab
        self.chunk_size = max_chunk

    def __len__(self):
        return (len(self.doc) - 1) // self.chunk_size

    def __getitem__(self, idx):
        start = idx * self.chunk_size
        substring = self.doc[start:start+self.chunk_size]
        x = self.vocab.lookup_indices(list(substring))
        substring = self.doc[start+1:start+self.chunk_size+1]
        y = self.vocab.lookup_indices(list(substring))
        return torch.tensor(x, dtype=torch.int), torch.tensor(y, dtype=torch.int)


class ARNet(L.LightningModule):
    def __init__(self, optim, num_embeds, embed_size, hidden_size, loss_func, layers=1):
        super().__init__()
        self.optim = optim
        self.hidden_size = hidden_size
        self.num_embeds = num_embeds
        self.embed_size = embed_size
        self.loss_func = loss_func
        self.layers = layers

        self.save_hyperparameters()

        self.embed_layer = nn.Embedding(self.num_embeds, self.embed_size)
        self.gru_layers = nn.ModuleList(
            [nn.GRUCell(self.embed_size, self.hidden_size)] +
            [nn.GRUCell(self.hidden_size, self.hidden_size) for _ in range(self.layers-1)]
        )
        self.norm_layers = nn.ModuleList(
            [nn.LayerNorm(self.hidden_size) for _ in range(self.layers)]
        )
        self.pred_block = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.num_embeds)
        )

    def make_hidden_states(self, B):
        return [torch.zeros(B, self.hidden_size, device=self.device) for _ in range(len(self.gru_layers))]

    def forward(self, x):
        B = x.size(0)
        T = x.size(1)

        x = self.embed_layer(x)
        hidden_states = self.make_hidden_states(B)
        last_activations = []
        for t in range(T):
            x_in = x[:, t, :]
            last_activations.append(self.time_step(x_in, hidden_states))
        last_activations = torch.stack(last_activations, dim=1)
        return last_activations

    def time_step(self, x_in, hidden_states):
        if len(x_in.shape) == 1:
            x_in = self.embed_layer(x_in)

        if hidden_states is None:
            hidden_states = self.make_hidden_states(x_in.shape[0])

        for l in range(len(self.gru_layers)):
            h_prev = hidden_states[l]
            h_new = self.gru_layers[l](x_in, h_prev)
            h_new = self.norm_layers[l](h_new)
            hidden_states[l] = h_new
            x_in = h_new
        return self.pred_block(x_in)

    def training_step(self, batch, idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_func(preds, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return self.optim(self.parameters())


class CrossEntropyTimeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y):
        T = x.size(1)
        loss = 0
        for t in range(T):
            loss += self.loss_func(x[:, t, :], y[:, t])
        return loss


if __name__ == '__main__':
    data = get_shakespear_data()
    vocab = build_vocab_from_iterator(list(data))

    ds = AutoRegressiveDataset(
        doc=data,
        vocab=vocab,
        max_chunk=250
    )
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=10)

    optim = partial(torch.optim.Adam, lr=0.001)
    loss_func = CrossEntropyTimeLoss()
    mod = ARNet(
        optim=optim,
        num_embeds=len(vocab),
        embed_size=128,
        hidden_size=128,
        loss_func=loss_func,
        layers=2
    )
    for p in mod.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -2, 2))

    # track experimental data by using Aim
    aim_logger = AimLogger(
        experiment="Character RNN",
        train_metric_prefix="train_",
    )

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="mps",
        devices=1,
        logger=aim_logger,
        log_every_n_steps=1,
        callbacks=[RichProgressBar(refresh_rate=5, leave=True)],
    )
    trainer.fit(mod, train_dataloaders=dl)

