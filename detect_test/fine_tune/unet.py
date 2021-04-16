from __future__ import print_function

import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from metrics import DiceCoefficient, DiceLoss
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCELoss, Sequential
from torch.optim import Adam

dir_path = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(dir_path, "img")
if not os.path.exists(img_path):
    os.makedirs(img_path)


def make_conv3d(inplane, plane, stride=1, kernel=(3, 3, 3), padding=1):
    return nn.Sequential(
        nn.Conv3d(inplane, plane, kernel, stride, padding),
        nn.InstanceNorm3d(plane),
        nn.ReLU()
    )


def make_transconv3d(inplane, plane, stride=2, kernel=(2, 2, 2)):
    return nn.Sequential(
        nn.ConvTranspose3d(inplane, plane, kernel, stride),
        nn.InstanceNorm3d(plane),
        nn.ReLU()
    )


class Unet(LightningModule):
    """
    为了有更快的训练速度, 降采样部分使用stride=2的max pooling
    """

    def __init__(self, channels=[1, 2, 4, 8, 16]):
        super(Unet, self).__init__()
        self.layer0 = nn.Sequential(
            make_conv3d(channels[0], channels[1]),  # 256
            make_conv3d(channels[1], channels[1])
        )

        self.layer1 = nn.Sequential(
            # make_conv3d(channels[1], channels[1], stride=2),  # 128
            nn.MaxPool3d(kernel_size=2, stride=2),
            make_conv3d(channels[1], channels[2])
        )

        self.layer2 = nn.Sequential(
            # make_conv3d(channels[2], channels[2], stride=2),  # 64
            nn.MaxPool3d(kernel_size=2, stride=2),
            make_conv3d(channels[2], channels[3])
        )

        self.layer3 = nn.Sequential(
            # make_conv3d(channels[3], channels[3], stride=2),  # 32
            nn.MaxPool3d(kernel_size=2, stride=2),
            make_conv3d(channels[3], channels[4])
        )

        self.up_conv0 = make_transconv3d(channels[4], channels[3])
        self.up_layer0 = nn.Sequential(
            make_conv3d(channels[4], channels[3]),
            make_conv3d(channels[3], channels[3])
        )

        self.up_conv1 = make_transconv3d(channels[3], channels[2])
        self.up_layer1 = nn.Sequential(
            make_conv3d(channels[3], channels[2]),
            make_conv3d(channels[2], channels[2])
        )

        self.up_conv2 = make_transconv3d(channels[2], channels[1])
        self.up_layer2 = nn.Sequential(
            make_conv3d(channels[2], channels[1]),
            make_conv3d(channels[1], channels[1])
        )

        self.conv = make_conv3d(channels[1], channels[0])
        self.activation = nn.Sigmoid()

        self.criterion = DiceLoss()
        self.bce = BCELoss()
        self.dice_coefficient = DiceCoefficient()

        # output file
        self.fp = open("output.txt", "w")

    def forward(self, x) -> Tensor:
        output0 = self.layer0(x)
        output1 = self.layer1(output0)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)

        output = self.up_conv0(output3)
        output = self.up_layer0(torch.cat([output2, output], dim=1))

        output = self.up_conv1(output)
        output = self.up_layer1(torch.cat([output1, output], dim=1))

        output = self.up_conv2(output)
        output = self.up_layer2(torch.cat([output0, output], dim=1))

        output = self.conv(output)
        output = self.activation(output)
        return output

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-4)
        return optim

    def training_step(self, batch, batch_idx):
        data, nodule = batch
        out = self(data)
        loss1 = self.criterion(out, nodule)
        loss2 = self.bce(out, nodule)
        loss = loss1 + loss2
        with torch.no_grad():
            dice = self.dice_coefficient(out, nodule)
        self.log_dict({"dice": dice.item()}, prog_bar=True)

        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        data, nodule = batch
        out = self(data)

        out = out[0][0].cpu()
        nodule = nodule[0][0].cpu()

        for idx, (ct, nod) in enumerate(zip(out, nodule)):
            fig, ax = plt.subplots(2, 1)
            ax[0].imshow(ct, cmap="bone")
            ax[0].axis("off")
            ax[1].imshow(nod, cmap="bone")
            ax[1].axis("off")
            plt.savefig(os.path.join(
                img_path, f"{idx}.png"), bbox_inches="tight")
            plt.close(fig)
