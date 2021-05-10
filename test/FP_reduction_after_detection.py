import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_path)

import torch
import numpy as np
from test.resnet import Resnet3D
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from glob import glob
from test.detect_data import DetectResData

def main():
    for fold in range(10):
        ckpt_path = os.path.join(save_path, "ckpt", f"{fold}")
        ckpt_list = glob(os.path.join(ckpt_path, "*.ckpt"))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            ckpt = ckpt_list[-1]
        else:
            ckpt = None

        log_path = os.path.join(save_path, "lightning_logs", f"{fold}")
        ckpt_model = ModelCheckpoint(ckpt_path, monitor="accuracy", mode="max", save_top_k=1)
        logger = TensorBoardLogger(log_path)

        model = Resnet3D(1, 1, dropout=DROPOUT, save_root=save_path)
        trainer = Trainer(logger=logger,
                          callbacks=[ckpt_model],
                          gpus=[1],
                          max_epochs=EPOCH,
                          benchmark=True,
                          resume_from_checkpoint=ckpt,
                          )

        data_module = DetectResData("result.csv", fold)

        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module, verbose=True)

if __name__ == "__main__":
    EPOCH = 50
    DROPOUT = 0.2
    save_path = os.path.join(project_path, "test", "reduction")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    main()