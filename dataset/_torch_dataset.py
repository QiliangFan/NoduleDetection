import os
from glob import glob
from typing import Sequence, Tuple

from torch.utils.data import Dataset
import torch
import numpy as np

class _BaseDataset(Dataset):

    def __init__(self, ct_list: Sequence, nodule_list: Sequence):
        super().__init__()

        self.ct_list = ct_list
        self.nodule_list = nodule_list
        assert len(self.ct_list) == len(self.nodule_list)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        ct, nodule = self.ct_list[index], self.nodule_list[index]
        ct, nodule = np.load(ct).astype(np.float32), np.load(nodule).astype(np.float32)
        ct, nodule = torch.from_numpy(ct), torch.from_numpy(nodule)
        ct.unsqueeze_(dim=0), nodule.unsqueeze_(dim=0)
        return ct, nodule

    def __len__(self):
        return len(self.ct_list)
