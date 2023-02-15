import os
import torch
import numpy as np

from torch.utils.data import Dataset
from random import shuffle


class ACIDS_Dataset(Dataset):
    def __init__(self, index_file):
        print("index_file: ", index_file)
        self.sample_files = list(np.loadtxt(index_file, dtype=str))

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_files[idx])
        data = sample["data"]
        label = sample["label"]
        
        return data, label
