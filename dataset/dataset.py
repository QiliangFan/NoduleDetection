from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from ._torch_dataset import _BaseDataset
from typing import Optional
import os
from glob import glob

class BaseDataset(LightningDataModule):

    def __init__(self, data_root: str, fold_idx: int):
        super().__init__()

        self.ct_root = os.path.join(data_root, "ct")
        self.nodule_root = os.path.join(data_root, "nodule")

        self.train_ct_list = []
        self.train_nodule_list = []

        self.test_ct_list = []
        self.test_nodule_list = []

        for i in range(10):
            ct_list = glob(os.path.join(self.ct_root, "**", f"subset{i}", "**", "*.npy"), recursive=True)
            if i != fold_idx:
                self.train_ct_list.extend(ct_list)
                self.train_nodule_list.extend([ct.replace(self.ct_root, self.nodule_root) for ct in ct_list])
            else:
                self.test_ct_list.extend(ct_list)
                self.test_nodule_list.extend([ct.replace(self.ct_root, self.nodule_root) for ct in ct_list])

    def prepare_data(self):
        self.train_data = _BaseDataset(self.train_ct_list, self.train_nodule_list)
        self.test_data = _BaseDataset(self.test_ct_list, self.test_nodule_list)
        self.train_data = DataLoader(self.train_data, batch_size=2, shuffle=True, num_workers=8, prefetch_factor=2)
        self.test_data = DataLoader(self.test_data, batch_size=2, shuffle=False, num_workers=8, prefetch_factor=2)
    
    def setup(self, stage: Optional[str]):
        print(f"Stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return self.train_data
    
    def test_dataloader(self) -> DataLoader:
        return self.test_data
