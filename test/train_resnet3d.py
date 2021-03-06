import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)
from test.data_module import Data, DataModule
from test.resnet import Resnet3D
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
import numpy as np
import torch.nn as nn
import torch
import json
import pickle
import random
from glob import glob
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import argparse
from torch.utils.data.dataset import ConcatDataset
from detect_test.data_module import UnetDataModule


def main():
    checkpoint_root = os.path.join(save_path, "checkpoints", "resnet3d")
    for i in range(FOLD):
        # 检查checkpoint是否存在
        if os.path.exists(f"{checkpoint_root}/fold{i}"):
            print(f"LOADING FOLD {i} checkpoints...")

        ckpt_dir = os.path.join(checkpoint_root, f"fold{i}")
        ckpt_files = os.listdir(ckpt_dir) if os.path.exists(ckpt_dir) else []
        ckpt_files.sort()

        if len(ckpt_files) > 0:
            print(f"Find ckpt {ckpt_files[-1]}")
            ckpt = os.path.join(f"{checkpoint_root}/fold{i}", ckpt_files[-1])
        else:
            print("No ckpt...")
            ckpt = None

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{checkpoint_root}/fold{i}", monitor="precision", mode="max", save_top_k=1)
        logger = TensorBoardLogger(
            f"{save_path}/resnet3d_logs/fold{i}", name="10-fold")

        epoch = 5
        # trainer = Trainer(gpus=[0 if run_name == "sub_last" else 1], logger=logger, callbacks=[
        #                   checkpoint_callback], max_epochs=epoch, resume_from_checkpoint=ckpt, benchmark=True)
        trainer = Trainer(gpus=1, logger=logger, callbacks=[
                          checkpoint_callback], max_epochs=epoch, resume_from_checkpoint=ckpt, benchmark=True)

        model = Resnet3D(in_channel=1, num_classes=1,
                         dropout=DROPOUT, save_root=save_path)

        # 先主要学正例，降采样负例

        data_module = DataModule(i, FOLD, tmp_aug_root, run_name)
        test_data_module = UnetDataModule(i, data_root=data_root, nodule_root=nodule_root, aug_root=aug_root, total_fold=10, batch_size=32)
        trainer.fit(model=model, datamodule=data_module)

        trainer.test(model=model, datamodule=test_data_module,
                     verbose=True)   # 两个data module的测试数据集都是一样的

        with open(os.path.join(save_path, "eval_metrics.txt"), "a") as fp:
            print(json.dumps(trainer.metrics_to_scalars(trainer.logged_metrics), indent=4), file=fp)


if __name__ == "__main__":
    # 48
    # tmp_pos_root = "/home/maling/fanqiliang/data/tmp/patch/1"
    # tmp_neg_root = "/home/maling/fanqiliang/data/tmp/patch/0"
    # tmp_aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"

    # 64
    tmp_pos_root = "/home/maling/fanqiliang/data/tmp64/patch/1"
    tmp_neg_root = "/home/maling/fanqiliang/data/tmp64/patch/0"
    tmp_aug_root = "/home/maling/fanqiliang/data/tmp64/augmented_data"

    aug_root = "/home/maling/fanqiliang/data/luna16/cube_aug"
    data_root = "/home/maling/fanqiliang/data/luna16/cube_ct"
    nodule_root = "/home/maling/fanqiliang/data/luna16/cube_nodule" 

    metric_file = os.path.join(project_path, "test", "resnet3d.json")
    # train = True

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ckpt", default=None, type=str)
    argparser.add_argument("--name", default="default", type=str)
    config = vars(argparser.parse_args())
    ckpt_path = config["ckpt"]
    run_name = config["name"]
    save_path = os.path.join(project_path, "test", run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    FOLD = 10

    DROPOUT = 0
    main()
