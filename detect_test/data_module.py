import os
from glob import glob
from typing import Sequence

import numpy as np
import SimpleITK as sitk
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class UnetDataModule(LightningDataModule):
    class Data(Dataset):
        def __init__(self, ct_list: Sequence[str], nodule_list: Sequence[str]):
            super(UnetDataModule.Data, self).__init__()
            self.data_list = ct_list
            self.nodule_list = nodule_list

        def __getitem__(self, idx):
            img = sitk.ReadImage(self.data_list[idx])
            arr: np.ndarray = sitk.GetArrayFromImage(img).astype(np.float32)
            arr = arr.reshape([1, *arr.shape])

            nodule = sitk.ReadImage(self.nodule_list[idx])
            nodule_arr = sitk.GetArrayFromImage(nodule).astype(np.float32)
            nodule_arr = nodule_arr.reshape([1, *nodule_arr.shape])
            return torch.as_tensor(arr), torch.as_tensor(nodule_arr)

        def __len__(self):
            return len(self.data_list)

    def __init__(self, fold_num: int,
                 data_root: str,
                 nodule_root: str,
                 total_fold: int = 10,
                 batch_size: int = 8):
        """
        一折作为验证集, 剩余9折作为训练集
        """
        super(UnetDataModule, self).__init__()
        assert f"expected fold_num to be smaller than {total_fold}, but got {fold_num}"
        self.batch_size = batch_size
        self.data_root = data_root
        self.nodule_root = nodule_root

        self.train_files = []
        self.test_files = []
        for i in range(total_fold):
            if i == fold_num:
                self.test_files.extend(
                    glob(os.path.join(data_root, f"subset{i}", "*.*")))
            else:
                self.train_files.extend(
                    glob(os.path.join(data_root, f"subset{i}", "*.*")))
        self.train_nodule_files = list(map(lambda x: x.replace(
            self.data_root, self.nodule_root), self.train_files))
        self.test_nodule_files = list(map(lambda x: x.replace(
            self.data_root, self.nodule_root), self.test_files))

    def setup(self, stage: str):
        if stage == "fit":  # train
            data = self.Data(self.train_files, self.train_nodule_files)
            self.data = DataLoader(
                data, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)
        elif stage == "test":  # test
            data = self.Data(self.test_files, self.test_nodule_files)
            self.data = DataLoader(
                data, batch_size=4, shuffle=True, pin_memory=True, num_workers=4)

    def train_dataloader(self):
        return self.data

    def test_dataloader(self):
        return self.data
