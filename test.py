import torch
from models.unet import Unet3D
from network import NetWork
from pytorch_lightning import Trainer

model = Unet3D(num_classes=1)
net = NetWork(model)
trainer = Trainer(max_epochs=10, benchmark=True, gpus=1)

from dataset import BaseDataset
data_root = "/home/maling/fanqiliang/data/luna16"
fold_idx = 5
data = BaseDataset(data_root, fold_idx)
trainer.fit(net, datamodule=data)