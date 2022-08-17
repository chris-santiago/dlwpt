import io
import string
import unicodedata
import zipfile
import re

import requests
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from dlwpt import ROOT

DATA_DIR = ROOT.joinpath("data")


def set_device():
    device = {True: torch.device("mps"), False: torch.device("cpu")}
    return device[torch.backends.mps.is_available()]


def get_mnist_datasets():
    train = MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
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

