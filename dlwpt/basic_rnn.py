import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dlwpt.utils import get_language_data, get_alphabet


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

    def forward(self, input):
        output, last_step = input[0], input[1]
        if isinstance(last_step, tuple):
            last_step = last_step[0]
        batch_size = last_step.shape[1]
        last_step = last_step.view(self.layers, self.num_directions, batch_size, -1)
        last_step = last_step[self.layers-1]
        last_step = last_step.permute(1, 0, 2)  # batch in first dim
        return last_step.reshape(batch_size, -1)


class RnnModel(nn.Module):
    def __init__(self, vocab_size, n_classes, embed_dims=64, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.embed_dims = embed_dims
        self.hidden_size = hidden_size
        self.loss_func = nn.CrossEntropyLoss()

        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.embed_dims),
            nn.RNN(self.embed_dims, self.hidden_size, batch_first=True),
            LastTimeStep(),
            nn.Linear(self.hidden_size, self.n_classes)
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    data = get_language_data(verbose=True)
    letters, alphabet = get_alphabet()
    dataset = LanguageNameDataset(data, alphabet)

    test_size = 300
    train, test = torch.utils.data.random_split(dataset, (len(dataset)-test_size, test_size))
    train_ldr = DataLoader(train, batch_size=1, shuffle=True)
    test_ldr = DataLoader(test, batch_size=1, shuffle=False)


