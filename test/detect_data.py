from __future__ import print_function

import torch
import numpy as np
from torch._C import TreeView
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
import pandas as pd
import os
from typing import Sequence

src = "/home/nku2/fanqiliang/data/luna16"
dst = "/home/maling/fanqiliang/data/luna16"

class Data(Dataset):

    def __init__(self, files: Sequence[str], preds: Sequence[float], targets: Sequence[float]):
        super().__init__()

        self.files = [f.replace(src, dst) for f in files]
        self.preds = preds
        self.targets = targets

    def __getitem__(self, idx):
        arr = np.load(self.files).astype(dtype=np.float32)
        arr = torch.as_tensor(arr)
        label = torch.as_tensor([self.targets[idx]])
        return arr, label

    def __len__(self):
        return len(self.files)

class DetectResData(LightningDataModule):

    def __init__(self, csv_file: str, fold_idx: int):
        super().__init__()

        assert os.path.exists(csv_file), f"{csv_file} must exists!"
        csv = pd.read_csv(csv_file, header=None)
        csv = csv[csv[1] > 0]  # 只取检测阶段预测为正例的样本

        self.files = csv[0].to_list()
        self.preds = csv[1].to_list()
        self.targets = csv[2].to_list()

        length = len(self.files)
        each_len = length // 10

        self.train_files = []
        self.train_preds = []
        self.train_targets = []

        self.test_files = []
        self.test_preds = []
        self.test_targets = []

        for i in range(10):
            if i != fold_idx:
                self.train_files.append(self.files[i*each_len:i*each_len+each_len])
                self.train_preds.append(self.preds[i*each_len:i*each_len+each_len])
                self.train_targets.append(self.targets[i*each_len:i*each_len+each_len])
            else:
                self.test_files.append(self.files[i*each_len:i*each_len+each_len])
                self.test_preds.append(self.preds[i*each_len:i*each_len+each_len])
                self.test_targets.append(self.targets[i*each_len:i*each_len+each_len])

    def prepare_data(self):
        self.train_data = Data(self.train_files, self.train_preds, self.train_targets)

        self.test_data = Data(self.test_files, self.test_preds, self.test_targets)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = DataLoader(self.train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)

            self.val_data = DataLoader(self.test_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)
        else:
            self.test_data = DataLoader(self.test_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)

    def train_dataloader(self) -> DataLoader:
        return self.train_data

    def val_dataloader(self) -> DataLoader:
        return self.val_data

    def test_dataload(self) -> DataLoader:
        return self.test_data


