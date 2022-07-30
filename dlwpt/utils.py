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
