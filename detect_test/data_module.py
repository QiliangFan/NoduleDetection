import os
from glob import glob
from typing import Sequence, List

import numpy as np
import SimpleITK as sitk
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


# augment: 复制数据的倍数
AUG_TIMES = 20
NUM_WORK = 8
class UnetDataModule(LightningDataModule):
    class Data(Dataset):
        def __init__(self, ct_list: Sequence[str], nodule_list: Sequence[str]):
            super(UnetDataModule.Data, self).__init__()
            self.data_list = ct_list
            self.nodule_list = nodule_list

        def __getitem__(self, idx):
            if self.data_list[idx].endswith(".mhd"):
                img = sitk.ReadImage(self.data_list[idx])
                arr: np.ndarray = sitk.GetArrayFromImage(
                    img).astype(np.float32)
                arr = arr.reshape([1, *arr.shape])

                nodule = sitk.ReadImage(self.nodule_list[idx])
                nodule_arr = sitk.GetArrayFromImage(nodule).astype(np.float32)
                nodule_arr = nodule_arr.reshape([1, *nodule_arr.shape])
            else:   # assume to be ".npy"
                arr: np.ndarray = np.load(
                    self.data_list[idx]).astype(np.float32)
                arr = arr.reshape((1, *arr.shape))
                nodule_arr: np.ndarray = np.load(
                    self.nodule_list[idx]).astype((np.float32))
                nodule_arr = nodule_arr.reshape((1, *nodule_arr.shape))
            label = 1 if np.any(nodule_arr > 0) else 0
            label_tensor = torch.zeros((2,))
            label_tensor[label] = 1
            # return self.data_list[idx], torch.as_tensor(arr), torch.as_tensor(label, dtype=torch.float32).unsqueeze(dim=0)
            return self.data_list[idx], torch.as_tensor(arr), label_tensor

        def __len__(self):
            return len(self.data_list)

    def __init__(self, fold_num: int,
                 data_root: str,
                 nodule_root: str,
                 aug_root: str,
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
        self.aug_root = aug_root

        self.train_files: List[str] = []
        self.test_files: List[str] = []
        for i in range(total_fold):
            if i == fold_num:
                self.test_files.extend(
                    glob(os.path.join(data_root, f"subset{i}", "*.*")))
            else:
                self.train_files.extend(
                    glob(os.path.join(data_root, f"subset{i}", "*.*")))

        # augmentation files
        self.train_files.extend(glob(os.path.join(self.aug_root, "**", "*.???")) * AUG_TIMES)

        self.train_nodule_files = list(map(lambda x: x.replace(
            self.data_root, self.nodule_root).replace(self.aug_root, self.nodule_root), self.train_files))
        self.test_nodule_files = list(map(lambda x: x.replace(
            self.data_root, self.nodule_root), self.test_files))

    def setup(self, stage: str):
        if stage == "fit":  # train
            data = self.Data(self.train_files, self.train_nodule_files)
            self.data = DataLoader(
                data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORK)

            data = self.Data(self.test_files, self.test_nodule_files)
            self.val_data = DataLoader(
                data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORK)

        elif stage == "test":  # test
            data = self.Data(self.test_files, self.test_nodule_files)
            self.data = DataLoader(
                data, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=NUM_WORK)

    def train_dataloader(self):
        return self.data

    def test_dataloader(self):
        return self.data

    def val_dataloader(self):
        return self.val_data