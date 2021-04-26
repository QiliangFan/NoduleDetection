import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, project_path)
from glob import glob

import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from network.LeNet import LeNet
# from test.data_module import DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from detect_test.data_module import UnetDataModule

# data_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
# nodule_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
# aug_root = "/home/fanrui/fanqiliang/data/luna16/cube_aug"


aug_root = "/home/maling/fanqiliang/data/luna16/cube_aug"
data_root = "/home/maling/fanqiliang/data/luna16/cube_ct"
nodule_root = "/home/maling/fanqiliang/data/luna16/cube_nodule"

def main():
    for fold in range(10):
        ckpt_path = os.path.join(dir_path, "ckpt", f"{fold}")
        ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
        logger = TensorBoardLogger(save_dir=save_path)
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt = ckpt_list[-1]
        else:
            ckpt = None
        model = LeNet((64, 64, 64), save_dir=save_path)
        model_ckpt = ModelCheckpoint(filepath=ckpt_path, monitor="recall")
        trainer = Trainer(gpus=[0], callbacks=[model_ckpt], max_epochs=50, resume_from_checkpoint=ckpt, logger=logger)

        # data_module = DataModule(fold, fold_num=10, aug_root=aug_root, run_name="lenet")
        data_module = UnetDataModule(fold, data_root=data_root, nodule_root=nodule_root, aug_root=aug_root, batch_size=32)
        trainer.fit(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(dir_path, "LeNet-5-detect")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    main()
