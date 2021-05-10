from __future__ import print_function

import torch
import numpy as np
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
        arr = np.load(self.files[idx]).astype(dtype=np.float32)
        arr = arr[16:47, 16:47, 16:47]
        arr = torch.as_tensor(arr).unsqueeze(dim=0)
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

        pos_csv = csv[csv[2] > 0]
        neg_csv = csv[csv[2] == 0]

        self.pos_files = pos_csv[0].to_list()
        self.pos_preds = pos_csv[1].to_list()
        self.pos_targets = pos_csv[2].to_list()

        self.neg_files = neg_csv[0].to_list()
        self.neg_preds = neg_csv[1].to_list()
        self.neg_targets = neg_csv[2].to_list()

        self.train_files = []
        self.train_preds = []
        self.train_targets = []

        self.test_files = []
        self.test_preds = []
        self.test_targets = []

        pos_len = len(self.pos_files)
        neg_len = len(self.neg_files)
        pos_len = pos_len // 10
        neg_len = neg_len // 10

        for i in range(10):
            if i != fold_idx:
                self.train_files.extend(self.pos_files[i*pos_len:i*pos_len+pos_len] * 10)
                self.train_preds.extend(self.pos_preds[i*pos_len:i*pos_len+pos_len] * 10)
                self.train_targets.extend(self.pos_targets[i*pos_len:i*pos_len+pos_len] * 10)

                self.train_files.extend(self.neg_files[i*neg_len:i*neg_len+neg_len])
                self.train_preds.extend(self.neg_preds[i*neg_len:i*neg_len+neg_len])
                self.train_targets.extend(self.neg_targets[i*neg_len:i*neg_len+neg_len]) 

            else:
                self.test_files.extend(self.pos_files[i*pos_len:i*pos_len+pos_len])
                self.test_preds.extend(self.pos_preds[i*pos_len:i*pos_len+pos_len])
                self.test_targets.extend(self.pos_targets[i*pos_len:i*pos_len+pos_len])

                self.test_files.extend(self.neg_files[i*neg_len:i*neg_len+neg_len])
                self.test_preds.extend(self.neg_preds[i*neg_len:i*neg_len+neg_len])
                self.test_targets.extend(self.neg_targets[i*neg_len:i*neg_len+neg_len])

        import random
        random.shuffle(self.train_files)
        random.shuffle(self.test_files)

    def prepare_data(self):
        self.train_data = Data(self.train_files, self.train_preds, self.train_targets)

        self.test_data = Data(self.test_files, self.test_preds, self.test_targets)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = DataLoader(self.train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)

            self.val_data = DataLoader(self.test_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=8, prefetch_factor=8)
        else:
            self.test_data = DataLoader(self.test_data, batch_size=128, shuffle=True,  pin_memory=True, num_workers=8, prefetch_factor=8)

    def train_dataloader(self) -> DataLoader:
        return self.train_data

    def val_dataloader(self) -> DataLoader:
        return self.val_data

    def test_dataloader(self) -> DataLoader:
        return self.test_data


