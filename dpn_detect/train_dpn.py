import os
import sys
from glob import glob
dir_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(dir_path)
sys.path.append(project_path)

import torch
import torch.nn as nn
from network.dpn import dpn131
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data_module import UnetDataModule




data_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
nodule_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
aug_root = "/home/fanrui/fanqiliang/data/luna16/cube_aug"


def main():
    for fold in range(10):
        data_module = UnetDataModule(fold,
                                     data_root=data_root,
                                     nodule_root=nodule_root,
                                     aug_root=aug_root)

        model = dpn131()
        ckpt_path = os.path.join(dir_path, "ckpt", f"{fold}")
        model_ckpt = ModelCheckpoint(dirpath=ckpt_path)
        ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt = ckpt_list[-1]
        else:
            ckpt = None

        trainer = Trainer(gpus=[1], callbacks=[model_ckpt],
                          max_epochs=50, resume_from_checkpoint=ckpt)

        trainer.fit(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module, verbose=True)


if __name__ == "__main__":
    main()
