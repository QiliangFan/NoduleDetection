import os
from glob import glob
import sys
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
from debug_utils import show

from data_module1 import UnetDataModule
from unet import Unet

dir_root = os.path.dirname(os.path.abspath(__file__))
# torch.set_num_threads(10)
# 210
# if os.environ["IP"].endswith("210"):
#     data_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
#     nodule_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
#     aug_root = "/home/fanrui/fanqiliang/data/luna16/cube_aug"


# 219
# if os.environ["IP"].endswith("219"):
#     data_root = "/home/fanqiliang_be/data/luna16/cube_ct"
#     nodule_root = "/home/fanqiliang_be/data/luna16/cube_nodule"

# 209
aug_root = "/home/nku2/fanqiliang/data/luna16/cube_aug"
data_root = "/home/nku2/fanqiliang/data/luna16/cube_ct"
nodule_root = "/home/nku2/fanqiliang/data/luna16/cube_nodule"


def main():
    for i in range(10):
        ckpt_path = os.path.join(dir_root, "unet_logs", f"fold{i}")
        ckpt = ModelCheckpoint(dirpath=ckpt_path, filename="unet", monitor="val_dice")
        ckpt_file = glob(os.path.join(ckpt_path, "*.ckpt"))
        ckpt_file = ckpt_file[0] if len(ckpt_file) > 0 else None
        if ckpt_file is not None:
            print('loading ckpt...')
        else:
            print("no ckpt...")
        trainer = Trainer(
            callbacks=[ckpt], resume_from_checkpoint=ckpt_file, max_epochs=30, gpus=[0 if cuda >= 0 else None])
        model = Unet()
        data_module = UnetDataModule(
            i, data_root, nodule_root, aug_root=aug_root, batch_size=batch_size)

        # train and test
        if stage == "train":
            trainer.fit(model, datamodule=data_module)

            trainer.test(model, datamodule=data_module, verbose=True)
        else:
            trainer.test(model, datamodule=data_module, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, required=True,
                        help="Model stage: train or test")
    parser.add_argument("--cuda", type=int, default=-1,
                        help="-1: no gpu; n(n >= 0): use gpu-n")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")

    config = vars(parser.parse_args())

    stage = config["stage"]
    cuda = config["cuda"]
    batch_size = config["batchsize"]
    batch_size = 16

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    main()
