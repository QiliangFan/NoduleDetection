import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MyDataset(Dataset):

    def __init__(self, mhd_files, nodule_files):
        super(MyDataset, self).__init__()

        self.mhd_files = mhd_files
        self.nodule_files = nodule_files

    def __getitem__(self, idx):
        import SimpleITK as sitk
        mhd_file, nodule_file = self.mhd_files[idx], self.nodule_files[idx]
        mhd, nodule = sitk.GetArrayFromImage(sitk.ReadImage(mhd_file)).astype(
            np.float32), np.load(nodule_file).astype(np.float32)
        mhd, nodule = torch.from_numpy(mhd), torch.from_numpy(nodule)
        mhd, nodule = mhd.unsqueeze(0), nodule.unsqueeze(0)
        return mhd, nodule

    def __len__(self):
        return len(self.mhd_files)


class IntegrateData(LightningDataModule):
    """
    subset*
    """

    def __init__(self,
                 mhd_root: str,
                 nodule_root: str,
                 fold_idx: int,
                 batch_size: int = 4):
        import os
        from glob import glob
        super(IntegrateData, self).__init__()
        assert os.path.exists(mhd_root) and os.path.exists(
            nodule_root), f"expected path to exist but failed to find"
        self.batch_size = batch_size

        self.mhd_files = glob(os.path.join(
            mhd_root, "**", "*.mhd"), recursive=True)
        self.nodule_files = [mhd.replace(mhd_root, nodule_root).replace(
            ".mhd", ".npy") for mhd in self.mhd_files]

        self.train_mhd = []
        self.test_mhd = []

        self.train_nodule = []
        self.test_nodule = []

        for mhd, nodule in zip(self.mhd_files, self.nodule_files):
            if f"subset{fold_idx}" in mhd:
                self.test_mhd.append(mhd)
                self.test_nodule.append(nodule)
            else:
                self.train_mhd.append(mhd)
                self.train_nodule.append(nodule)

        self.train_data = MyDataset(self.train_mhd, self.train_nodule)
        self.test_data = MyDataset(self.test_mhd, self.test_nodule)

    def prepare_data(self):
        print(f"train files: {len(self.train_nodule)}")
        print(f"test files: {len(self.test_nodule)}")

    def setup(self, stage: str):
        """
        stage: fit | test
        """
        print(f"Stage: \033[33m {stage} \033[0m ...")

    def train_dataloader(self) -> DataLoader:
        data = DataLoader(self.train_data, batch_size=self.batch_size, pin_memory=True, num_workers=4, shuffle=True)
        return data

    def test_dataloader(self) -> DataLoader:
        data = DataLoader(self.test_data, batch_size=self.batch_size, pin_memory=True, num_workers=4, shuffle=False)
        return data