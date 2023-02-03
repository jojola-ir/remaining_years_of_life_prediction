import os
from os.path import join

import torch
import torch.nn as nn

from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode

        self.data = torch.load(join(data_dir, f'{mode}.pt'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]