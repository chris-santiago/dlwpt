import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from dlwpt.trainer import Trainer
from dlwpt.utils import get_language_data, get_alphabet, set_device, pad_and_pack


class LanguageNameDataset(Dataset):
    def __init__(self, lang_names, vocab):
        self.label_names = list(lang_names.keys())
        self.data = []
        self.labels = []
        self.vocab = vocab

        for label, lang in enumerate(self.label_names):
            for sample in lang_names[lang]:
                self.data.append(sample)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def str_to_input(self, input_string):
        names = torch.zeros(len(input_string), dtype=torch.long)
        for pos, char in enumerate(input_string):
            names[pos] = self.vocab[char]
        return names

    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        return self.str_to_input(name), label


class LastTimeStep(nn.Module):
    def __init__(self, layers=1, bidirectional=False):
        super().__init__()
        self.layers = layers
        self.num_directions = 2 if bidirectional else 1

    def forward(self, inputs):
        _, last_step = inputs[0], inputs[1]
        if isinstance(last_step, tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1]
        last_step = last_step.view(self.layers, self.num_directions, batch_size, -1)
        last_step = last_step[self.layers - 1]
        last_step = last_step.permute(1, 0, 2)  # batch in first dim
        return last_step.reshape(batch_size, -1)


class EmbeddingPackable(nn.Module):
    def __init__(self, embed_layer):
        super().__init__()
        self.embed_layer = embed_layer

    def forward(self, inputs):
        if type(inputs) == torch.nn.utils.rnn.PackedSequence:
            sequences, lengths = pad_packed_sequence(inputs, batch_first=True)
            sequences = self.embed_layer(sequences)
            return pack_padded_sequence(
                sequences, lengths, batch_first=True, enforce_sorted=False
            )
        else:
            return self.embed_layer(inputs)


class RnnModel(nn.Module):
    def __init__(
        self, vocab_size, n_classes, embed_dims=64, hidden_size=256, optim=None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embed_dims = embed_dims
        self.hidden_size = hidden_size
        self.loss_func = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            EmbeddingPackable(nn.Embedding(self.vocab_size, self.embed_dims)),
            nn.RNN(
                self.embed_dims, self.hidden_size, batch_first=True, bidirectional=True
            ),
            LastTimeStep(),
            nn.Linear(self.hidden_size * 2, self.n_classes),
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        logits = self(x)
        return F.softmax(logits, dim=-1)


if __name__ == "__main__":
    from dlwpt import ROOT
    from datetime import datetime

    NOW = datetime.now().strftime("%Y%m%d-%H%M")
    LOG_DIR = ROOT.joinpath("runs", NOW)

    data = get_language_data(verbose=True)
    letters, alphabet = get_alphabet()
    dataset = LanguageNameDataset(data, alphabet)

    test_size = 300
    bs = 16
    train, test = torch.utils.data.random_split(
        dataset, (len(dataset) - test_size, test_size)
    )
    train_ldr = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=pad_and_pack)
    test_ldr = DataLoader(test, batch_size=bs, shuffle=False, collate_fn=pad_and_pack)

    device = set_device()
    mod = RnnModel(vocab_size=len(letters), n_classes=len(dataset.label_names))
    trainer = Trainer(mod, epochs=20, device=device, log_dir=LOG_DIR)
    trainer.fit(train_ldr, test_ldr)
