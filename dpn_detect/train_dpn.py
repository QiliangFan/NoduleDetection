import os
import sys
from glob import glob
dir_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(dir_path)
sys.path.append(project_path)

from data_module import UnetDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from network.dpn import getdpn
import torch.nn as nn
import torch
from pytorch_lightning.loggers import TensorBoardLogger


# data_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
# nodule_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
# aug_root = "/home/fanrui/fanqiliang/data/luna16/cube_aug"

# aug_root = "/home/maling/fanqiliang/data/luna16/cube_aug"
# data_root = "/home/maling/fanqiliang/data/luna16/cube_ct"
# nodule_root = "/home/maling/fanqiliang/data/luna16/cube_nodule"

# 220
aug_root = "/home/maling/fanqiliang/data/luna16/cube16_aug"
data_root = "/home/maling/fanqiliang/data/luna16/cube16_ct"
nodule_root = "/home/maling/fanqiliang/data/luna16/cube16_nodule"

def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"dpn{v}")
    for fold in range(10):
        data_module = UnetDataModule(fold,
                                     data_root=data_root,
                                     nodule_root=nodule_root,
                                     aug_root=aug_root,
                                     batch_size=256)

        model = getdpn(save_dir=save_dir)
        ckpt_path = os.path.join(save_dir, "ckpt", f"{fold}")
        logger = TensorBoardLogger(save_dir=os.path.join(save_dir, "lightning_logs"))
        model_ckpt = ModelCheckpoint(dirpath=ckpt_path, monitor="recall", mode="max", save_top_k=1)
        ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt = ckpt_list[-1]
        else:
            ckpt = None

        trainer = Trainer(gpus=[v], callbacks=[model_ckpt],
                          max_epochs=50, resume_from_checkpoint=ckpt, benchmark=True, logger=logger)

        trainer.fit(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module, verbose=True)


if __name__ == "__main__":
    v = 0
    main()
