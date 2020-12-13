from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np


class Luna16Dataset(Dataset):

    def __init__(self,
                 data_root:str,
                 seg_root: str,
                 nodule_root: str,
                 sz_root: str,
                 weight: str):
        super(Luna16Dataset, self).__init__()
        # instance variable
        self.ct_list = []
        self.seg_list = []
        self.nodule_list = []
        self.sz_list = []
        self.weight_list = []

        for _ in os.listdir(data_root):
            self.ct_list.append(os.path.join(data_root, _))
            self.seg_list.append(os.path.join(seg_root, _))
            self.nodule_list.append(os.path.join(nodule_root, _))
            self.sz_list.append(os.path.join(sz_root, _))
            self.weight_list.append(os.path.join(weight, _))

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ct = np.load(self.ct_list[item])
        seg = np.load(self.seg_list[item])
        nodule = np.load(self.nodule_list[item])
        sz = np.load(self.sz_list[item])
        weight = np.load(self.weight_list[item])
        ct = torch.as_tensor(ct, dtype=torch.float32)
        seg = torch.as_tensor(seg, dtype=torch.long)
        nodule = torch.as_tensor(nodule, dtype=torch.long)
        sz = torch.as_tensor(sz, dtype=torch.float32)
        weight = torch.as_tensor(weight, dtype=torch.float32)
        idx = torch.where(seg == 1)
        z_max, z_min = torch.max(idx[0]), torch.min(idx[0])
        y_max, y_min = torch.max(idx[1]), torch.min(idx[1])
        x_max, x_min = torch.max(idx[2]), torch.min(idx[2])
        ct = ct[z_min:z_max, y_min:y_max, x_min:x_max]
        seg = seg[z_min:z_max, y_min:y_max, x_min:x_max]
        nodule = nodule[z_min:z_max, y_min:y_max, x_min:x_max]
        sz = sz[:, z_min:z_max, y_min:y_max, x_min:x_max]
        weight = weight[z_min:z_max, y_min:y_max, x_min:x_max]
        return ct, seg, nodule, sz, weight

    def __len__(self):
        return len(self.ct_list)