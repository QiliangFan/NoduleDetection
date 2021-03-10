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



def main():
    for i in range(FOLD):
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{save_path}/checkpoints/resnet3d/fold{i}")
        logger = TensorBoardLogger(
            f"{save_path}/resnet3d_logs/fold{i}", name="10-fold")

        trainer = Trainer(gpus=[0 if run_name == "sub_last" else 1], logger=logger, callbacks=[
                          checkpoint_callback], max_epochs=30, reload_dataloaders_every_epoch=True)

        model = Resnet3D(in_channel=1, num_classes=1,
                         dropout=DROPOUT, save_root=save_path)

        # 先主要学正例，降采样负例

        if run_name == "sub_last":
            data_module = DataModule(i, FOLD, aug_root, is_subsample=False)
            trainer.fit(model=model, datamodule=data_module)
            print("finished...")
            data_module_subsample = DataModule(
                i, FOLD, aug_root, is_subsample=True)
            trainer.fit(model=model, datamodule=data_module_subsample)
            print("finished...")
        else:
            data_module_subsample = DataModule(
                i, FOLD, aug_root, is_subsample=True)
            trainer.fit(model=model, datamodule=data_module_subsample)
            print("finished...")
            data_module = DataModule(i, FOLD, aug_root, is_subsample=False)
            trainer.fit(model=model, datamodule=data_module)
            print("finished...")

        trainer.test(model=model, datamodule=data_module_subsample,
                     verbose=True)   # 两个data module的测试数据集都是一样的

        with open(os.path.join(save_path, "eval_metrics.txt"), "a") as fp:
            print(json.dumps(trainer.logged_metrics, indent=4), file=fp)


if __name__ == "__main__":
    pos_root = "/home/maling/fanqiliang/data/tmp/patch/1"
    neg_root = "/home/maling/fanqiliang/data/tmp/patch/0"
    aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"

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

    DROPOUT = 0.2
    main()
