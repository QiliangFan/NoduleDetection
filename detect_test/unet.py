from __future__ import print_function
from pytorch_lightning import LightningModule
from torch.optim import Adam
import torch.nn as nn
from torch.nn import Sequential


def make_conv3d(inplane, outplane, stride=1, kernel=(3, 3, 3), padding=1):
    return Sequential(
        nn.Conv3d(inplane, outplane, stride=stride,
                  kernel_size=kernel, padding=padding),
        nn.InstanceNorm3d(outplane),
        nn.ReLU()
    )


def make_trans_conv3d(inplane, outplane, stride=1, kernel=(2, 2, 2)):
    return Sequential(
        nn.ConvTranspose3d(inplane, outplane,
                           kernel_size=kernel, stride=stride),
        nn.InstanceNorm3d(outplane),
        nn.ReLU()
    )


class Unet(LightningModule):
    """
    为了有更快的训练速度, 降采样部分使用stride=2的max pooling
    """

    def __init__(self, inplane, class_num=2):
        super(Unet, self).__init__()
        self.layer1 = nn.Sequential(
            make_conv3d(inplane, 10),
            make_conv3d(10, 10),
        )

        self.layer1 = nn.Sequential(
            nn.MaxPool3d((2, 2, 2, )),
            make_conv3d()
        )

    def forward(self):
        pass

    def configure_optimizers(self):
        optim = Adam(self.parameters(), weight_decay=1e-4)
        return optim

    def training_step(self):
        pass

    def test_step(self):
        pass
