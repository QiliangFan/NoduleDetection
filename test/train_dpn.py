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
from detect_test.data_module import UnetDataModule
from data_module import DataModule
from network.dpn import getdpn

dir_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(dir_path)


def main():
    for fold in range(10):
        ckpt_path = os.path.join(dir_path, "ckpt_dpn", f"{fold}")
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        model_ckpt = ModelCheckpoint(dirpath=ckpt_path, monitor="recall", verbose=True, save_top_k=1, mode="max")
        ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt = ckpt_list[-1]
            print("loading...")
        else:
            ckpt = None
            if stage == "test":
                exit(0)

        # data_module = DataModule(fold, 10, aug_root, "dpn")
        data_module = UnetDataModule(fold, data_root=data_root, nodule_root=nodule_root, aug_root=aug_root, total_fold=10, batch_size=32)
        trainer = Trainer(gpus=[0], callbacks=[model_ckpt], max_epochs=34, resume_from_checkpoint=ckpt)
        model = getdpn(save_dir=os.path.join(dir_path, "dpn"))

        trainer
        if stage == "train":
            trainer.fit(model, datamodule=data_module)
            # trainer.fit(model)
            trainer.test(model, datamodule=data_module, verbose=True)
        else:
            trainer.test(model, datamodule=data_module, verbose=True, ckpt_path=ckpt)

if __name__ == "__main__":
    # 220
    pos_root = "/home/maling/fanqiliang/data/tmp/patch/1"
    neg_root = "/home/maling/fanqiliang/data/tmp/patch/0"
    aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"

    aug_root = "/home/maling/fanqiliang/data/luna16/cube_aug"
    data_root = "/home/maling/fanqiliang/data/luna16/cube_ct"
    nodule_root = "/home/maling/fanqiliang/data/luna16/cube_nodule" 

    # 209
    # pos_root = "/home/nku2/fanqiliang/data/tmp/patch/1"
    # neg_root = "/home/nku2/fanqiliang/data/tmp/patch/0"
    # aug_root = "/home/nku2/fanqiliang/data/tmp/augmented_data"

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="train", type=str)
    config = vars(parser.parse_args())
    stage = config["stage"]
    main()
