from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
import numpy as np
import os
import sys
import json
from glob import glob
import random

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Data(Dataset):
    def __init__(self, data, label):
        super(Data, self).__init__()
        self.label = torch.as_tensor(label, dtype=torch.float32)
        if isinstance(data, str):
            self.files = glob(os.path.join(data, "*.npy"))
        elif isinstance(data, list):
            self.files = data.copy()
        else:
            sys.exit(1)

    def __getitem__(self, idx):
        file = self.files[idx]
        arr = np.load(file).astype(np.float32)
        arr = torch.from_numpy(arr)
        arr.unsqueeze_(dim=0)
        label = torch.ones((1,))
        return arr, label * self.label

    def __len__(self):
        return len(self.files)


class DataModule(LightningDataModule):
    def __init__(self, i: int, fold_num: int, aug_root: str, run_name):
        """
        i: i-th fold
        fold_num: total number of folds
        """
        super().__init__()
        with open(os.path.join(project_path, "test", "data.json"), "rb") as fp:
            DATA_JSON = json.load(fp)
        self.test_pos_files = DATA_JSON["positive"][i]
        self.test_neg_files = DATA_JSON["negative"][i]
        self.train_pos_files = []
        self.train_neg_files = []
        [[self.train_pos_files.extend(DATA_JSON["positive"][k]), self.train_neg_files.extend(DATA_JSON["negative"][k])]
            for k in range(fold_num) if k != i]
        aug_files = glob(os.path.join(aug_root, "*.npy"))
        self.train_pos_files.extend(aug_files)

        self.run_name = run_name
        # subsample
        self.sub_train_neg_files = random.sample(
            self.train_neg_files, len(self.train_neg_files)//5)
        print(f"Pos nums: {len(self.train_pos_files)}; Sub_neg nums: {len(self.sub_train_neg_files)}")

    def setup(self, stage):
        print(f"datamodule stage: ", stage)
        if stage in "fit":
            if self.run_name == "sub_first":
                self.train_pos_data_0 = Data(self.train_pos_files, label=1)
                self.train_neg_data_0 = Data(self.sub_train_neg_files, label=0)

                self.train_pos_data_1 = Data(self.train_pos_files, label=1)
                self.train_neg_data_1 = Data(self.train_neg_files, label=0)
            else:
                self.train_pos_data_0 = Data(self.train_pos_files, label=1)
                self.train_neg_data_0 = Data(self.train_neg_files, label=0)

                self.train_pos_data_1 = Data(self.train_pos_files, label=1)
                self.train_neg_data_1 = Data(self.sub_train_neg_files, label=0)
        elif stage in "test":
            self.test_pos_data = Data(self.test_pos_files, label=1)
            self.test_neg_data = Data(self.test_neg_files, label=0)

    def train_dataloader(self) -> DataLoader:
        print("train dataloader")
        # train_data = ConcatDataset(
        #     [self.train_pos_data_0, self.train_neg_data_0, self.train_pos_data_1, self.train_neg_data_1])
        train_data = ConcatDataset([
                                    self.train_pos_data_1, 
                                    self.train_neg_data_1, 
                                    self.train_pos_data_0
                                    ])
        # train_data = self.train_pos_data
        train_data = DataLoader(
            train_data, batch_size=32, pin_memory=True, num_workers=4, shuffle=True)
        return train_data

    def test_dataloader(self) -> DataLoader:
        print("test dataloader")
        test_data = ConcatDataset([self.test_pos_data, self.test_neg_data])
        # test_data = self.test_pos_data
        test_data = DataLoader(test_data, batch_size=32,
                               pin_memory=True, num_workers=4, shuffle=True)
        return test_data
