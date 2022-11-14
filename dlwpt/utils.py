import io
import string
import unicodedata
import zipfile
import re

import requests
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torchvision.datasets import MNIST
from torchvision.transforms import transforms, Compose

from dlwpt import ROOT

DATA_DIR = ROOT.joinpath("data")


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]


def get_mnist_datasets(do_augment=False):
    transform = {
        True: Compose([
            transforms.RandomAffine(degrees=5, translate=(.05, .05), scale=(.98, 1.02)),
            transforms.ToTensor()
        ]),
        False: transforms.ToTensor()
    }
    train = MNIST(DATA_DIR, train=True, download=True, transform=transform[do_augment])
    test = MNIST(DATA_DIR, train=False, download=True, transform=transforms.ToTensor())
    return train, test


def get_alphabet():
    letters = string.ascii_letters + ".,;'"
    alphabet = {v: k for k, v in dict(enumerate(letters)).items()}
    return letters, alphabet


def to_ascii(s):
    letters, _ = get_alphabet()
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in letters
    )


def get_language_data(verbose=False):
    url = 'https://download.pytorch.org/tutorial/data.zip'
    resp = requests.get(url)
    contents = zipfile.ZipFile(io.BytesIO(resp.content))
    contents.extractall()

    data = {}

    for fp in contents.namelist():
        if 'data/names/' in fp and fp.endswith('.txt'):
            lang = re.search(r'([A-Z]+)\w+', fp)[0]
            with contents.open(fp) as file:
                lang_names = [
                    to_ascii(line).lower()
                    for line in str(file.read(), encoding='utf-8').strip().split('\n')
                ]
                data[lang] = lang_names
            if verbose:
                print(f'{lang}: {len(lang_names)}')

    return data


def pad_and_pack(batch):
    inputs = []
    labels = []
    lengths = []
    for x, y in batch:
        inputs.append(x)
        labels.append(y)
        lengths.append(x.shape[0])
    x_padded = pad_sequence(inputs, batch_first=False)
    x_packed = pack_padded_sequence(x_padded, lengths, batch_first=False, enforce_sorted=False)
    y_batched = torch.as_tensor(labels, dtype=torch.long)
    return x_packed, y_batched


def add_noise(x, loc=0, scale=1):
    noise = torch.distributions.Normal(loc, scale)
    return x + noise.sample(sample_shape=x.shape)


class WhiteNoise(nn.Module):
    def __init__(self, loc=0, scale=1):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        return add_noise(x, self.loc, self.scale)
