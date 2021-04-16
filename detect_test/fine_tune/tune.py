import os
from glob import glob
from typing import Any, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from unet import Unet

nodule_train = "/home/fanrui/fanqiliang/data/luna16/cube_nodule/subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.323408652979949774528873200770_74.npy"
ct_train = "/home/fanrui/fanqiliang/data/luna16/cube_ct/subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.323408652979949774528873200770_74.npy"

nodule_test = "/home/fanrui/fanqiliang/data/luna16/cube_nodule/subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.290135156874098366424871975734_58.npy"
ct_test = "/home/fanrui/fanqiliang/data/luna16/cube_ct/subset5/1.3.6.1.4.1.14519.5.2.1.6279.6001.290135156874098366424871975734_58.npy"

dir_path = os.path.dirname(os.path.abspath(__file__))


class Data(Dataset):

    def __init__(self, nodule, ct):
        self.nodule = nodule
        self.ct = ct

    def __getitem__(self, idx):
        nodule = np.load(self.nodule).astype(np.float32)
        ct = np.load(self.ct).astype(np.float32)
        nodule, ct = torch.from_numpy(nodule), torch.from_numpy(ct)
        nodule, ct = nodule.unsqueeze(dim=0), ct.unsqueeze(dim=0)
        return nodule, ct

    def __len__(self):
        return 1


class DataModule(LightningDataModule):

    def __init__(self):
        super(DataModule, self).__init__()

    def prepare_data(self):
        pass

    def setup(self, stage: str):
        """
        stage: fit|test
        """
        if stage == "fit":
            self.train_data = Data(nodule_train, ct_train)

        else:
            self.test_data = Data(nodule_test, ct_test)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=1, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=1, num_workers=1)


def main():
    data_module = DataModule()

    model = Unet()

    ckpt_path = os.path.join(dir_path, "ckpt")
    model_ckpt = ModelCheckpoint(dirpath=ckpt_path)
    ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
    if len(ckpt_list) > 0:
        ckpt_list.sort()
        ckpt = ckpt_list[-1]
    else:
        ckpt = None

    trainer = Trainer(gpus=[1], callbacks=[model_ckpt], resume_from_checkpoint=ckpt, max_epochs=10000)

    trainer.fit(model, datamodule=data_module)

    trainer.test(model, datamodule=data_module)
    


if __name__ == "__main__":
    main()
