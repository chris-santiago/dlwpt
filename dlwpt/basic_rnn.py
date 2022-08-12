import torch
from torch.utils.data import Dataset

from dlwpt.utils import get_language_data


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
            names[pos] = self.vocab(char)
        return names
    
    def __getitem__(self, idx):
        name = self.data[idx]
        label = self.labels[idx]
        return self.string_to_input(name), label