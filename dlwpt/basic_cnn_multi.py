import torch
import torch.nn as nn


model = nn.parallel.DistributedDataParallel()


if __name__ == "__main__":
    devices = [torch.device("mps", index=i) for i in range(4)]
    print(devices)
