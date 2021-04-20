import torch
import torch.nn as nn
import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(project_path)
from glob import glob
import argparse
from data_module import Data, DataModule
from network.dpn import getdpn

dir_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(dir_path)


def main():
    for fold in range(10):
        ckpt_path = os.path.join(dir_path, "ckpt_dpn", f"{fold}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model_ckpt = ModelCheckpoint(dirpath=ckpt_path)
        ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt = ckpt_list[-1]
            print("loading...")
        else:
            ckpt = None
            if stage == "test":
                exit(0)

        data_module = DataModule(fold, 10, aug_root, "dpn")
        trainer = Trainer(gpus=[0], callbacks=[model_ckpt], max_epochs=5, resume_from_checkpoint=ckpt)
        model = getdpn(save_dir=os.path.join(dir_path, "dpn"))

        if stage == "train":
            trainer.fit(model, datamodule=data_module)
        else:
            trainer.test(model, datamodule=data_module, verbose=True)


if __name__ == "__main__":
    # 220
    # pos_root = "/home/maling/fanqiliang/data/tmp/patch/1"
    # neg_root = "/home/maling/fanqiliang/data/tmp/patch/0"
    # aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"

    # 209
    pos_root = "/home/nku2/fanqiliang/data/tmp/patch/1"
    neg_root = "/home/nku2/fanqiliang/data/tmp/patch/0"
    aug_root = "/home/nku2/fanqiliang/data/tmp/augmented_data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="train", type=str)
    config = vars(parser.parse_args())
    stage = config["stage"]
    main()
