from __future__ import print_function

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential
from torch.optim import Adam
from torch.nn import BCELoss
from typing import List

from metrics import DiceLoss, DiceCoefficient, AverageMeter, AccMeter


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
            make_conv3d(channels[1], channels[2], stride=2),  # 128
            make_conv3d(channels[2], channels[2])
        )

        self.layer2 = nn.Sequential(
            make_conv3d(channels[2], channels[3], stride=2),  # 64
            make_conv3d(channels[3], channels[3])
        )

        self.layer3 = nn.Sequential(
            make_conv3d(channels[3], channels[4], stride=2),  # 32
            make_conv3d(channels[4], channels[4])
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

        # criterion 
        self.criterion = DiceLoss()
        self.bce = BCELoss()
        self.dice_coefficient = DiceCoefficient()

        self.tp_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fn_meter = AverageMeter()

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
        optim = Adam(self.parameters(), lr=1e-3)
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

    def validation_step(self, batch, batch_idx):
        return self.inference(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.inference(batch, batch_idx, save=True)

    def test_epoch_end(self, outputs: List) -> None:
        metrics = self.trainer.logged_metrics
        for key in metrics:
            if isinstance(metrics[key], torch.Tensor):
                metrics[key] = metrics[key].item()
        
        import json
        with open("metrics.txt", "a") as fp:
            print(json.dumps(metrics, indent=4), file=fp)

    @torch.no_grad()
    def predict(self, batch, batch_idx):
        data, nodule = batch
        return self(data)

    @torch.no_grad()
    def inference(self, batch, batch_idx, save=False):
        data, nodule = batch
        out = self(data)
        dice = self.dice_coefficient(out, nodule)
        self.log_dict({"val_dice": dice}, prog_bar=True)
        if save:
            pass
    
        return batch_idx
