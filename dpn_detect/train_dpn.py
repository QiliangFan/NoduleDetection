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



# data_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
# nodule_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
# aug_root = "/home/fanrui/fanqiliang/data/luna16/cube_aug"

# aug_root = "/home/maling/fanqiliang/data/luna16/cube_aug"
# data_root = "/home/maling/fanqiliang/data/luna16/cube_ct"
# nodule_root = "/home/maling/fanqiliang/data/luna16/cube_nodule"

# 209
aug_root = "/home/nku2/fanqiliang/data/luna16/cube_aug"
data_root = "/home/nku2/fanqiliang/data/luna16/cube_ct"
nodule_root = "/home/nku2/fanqiliang/data/luna16/cube_nodule"

def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dpn")
    for fold in range(10):
        data_module = UnetDataModule(fold,
                                     data_root=data_root,
                                     nodule_root=nodule_root,
                                     aug_root=aug_root)

        model = getdpn(save_dir=save_dir)
        ckpt_path = os.path.join(dir_path, "ckpt", f"{fold}")
        model_ckpt = ModelCheckpoint(dirpath=ckpt_path, monitor="acc")
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
