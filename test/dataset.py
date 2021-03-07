from typing import List
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import numpy as np


class Data(Dataset):
    def __init__(self, data_list: List[str], label: int):
        super(Data, self).__init__()
        self.data = []
        self.data.extend(data_list)
        random.shuffle(self.data)
        self.label = torch.as_tensor(label, dtype=torch.float32)
    
    def __getitem__(self, idx):
        file = self.data[idx]
        arr = np.load(file).astype(np.float32)
        arr = torch.from_numpy(arr)
        return arr, self.label

    def __len__(self):
        return len(self.data)

