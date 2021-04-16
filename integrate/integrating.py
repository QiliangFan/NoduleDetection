import os
from glob import glob

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from feature_extraction import IntegrateData, ResUnet

root = os.path.dirname(os.path.abspath(__file__))

ct_mhd_root = "/home/fanrui/fanqiliang/lung16/LUNG16"
save_root = "/home/fanrui/fanqiliang/data/luna16/1_4_nodule"

def main():
    for fold in range(10):  # 10 fold
        ckpt_dir = os.path.join(root, "ckpt", f"{fold}")
        checkpoint = ModelCheckpoint(dirpath=ckpt_dir, verbose=True)
        ckpts = glob(os.path.join(ckpt_dir))
        if len(ckpts) > 0:
            ckpts.sort()
            ckpt = ckpts[-1]
        else:
            ckpt = None
        trainer = Trainer(gpus=[0], callbacks=[checkpoint], resume_from_checkpoint=ckpt)

        model = ResUnet(1)

        data_module = IntegrateData(ct_mhd_root, save_root, fold_idx=fold, batch_size=1)  # batch=1 -> ct size varies
        
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()
