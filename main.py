import torch
from models.unet import Unet3D
from network import NetWork
from pytorch_lightning import Trainer

def main():
    model = Unet3D()
    net = NetWork(model)
    trainer = Trainer(max_epochs=1, benchmark=True, gpus=1)
    


if __name__ == "__main__":
    main()