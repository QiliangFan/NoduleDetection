from __future__ import print_function

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential
from torch.optim import Adam
from torch.nn import BCELoss
from typing import List
from debug_utils import show

from metrics import DiceLoss, DiceCoefficient, AverageMeter, AccMeter


def make_conv3d(inplane, plane, stride=1, kernel=(3, 3, 3), padding=1):
    return nn.Sequential(
        nn.Dropout3d(),
        nn.Conv3d(inplane, plane, kernel, stride, padding, bias=False),
        # nn.BatchNorm3d(plane, eps=1e-3),
        nn.LocalResponseNorm(2),
        nn.ReLU(inplace=True)
    )


def make_transconv3d(inplane, plane, stride=2, kernel=(2, 2, 2)):
    return nn.Sequential(
        nn.Dropout3d(),
        nn.ConvTranspose3d(inplane, plane, kernel, stride, bias=False),
        nn.LocalResponseNorm(2),
        nn.ReLU(inplace=True)
    )


class Unet(LightningModule):
    """
    为了有更快的训练速度, 降采样部分使用stride=2的max pooling
    """

    def __init__(self, channels=[1, 2, 4, 8, 16]):
        super(Unet, self).__init__()
        self.in_channel = 1
        self.n_classes = 1

        self.ec0 = self.encoder(self.in_channel, 32, bias=False, batchnorm=False)
        self.ec1 = self.encoder(32, 64, bias=False, batchnorm=False)
        self.ec2 = self.encoder(64, 64, bias=False, batchnorm=False)
        self.ec3 = self.encoder(64, 128, bias=False, batchnorm=False)
        self.ec4 = self.encoder(128, 128, bias=False, batchnorm=False)
        self.ec5 = self.encoder(128, 256, bias=False, batchnorm=False)
        self.ec6 = self.encoder(256, 256, bias=False, batchnorm=False)
        self.ec7 = self.encoder(256, 512, bias=False, batchnorm=False)

        self.pool0 = nn.MaxPool3d(2)
        self.pool1 = nn.MaxPool3d(2)
        self.pool2 = nn.MaxPool3d(2)

        self.dc9 = self.decoder(512, 512, kernel_size=2, stride=2, bias=False)
        self.dc8 = self.decoder(256 + 512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc7 = self.decoder(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc6 = self.decoder(256, 256, kernel_size=2, stride=2, bias=False)
        self.dc5 = self.decoder(128 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc4 = self.decoder(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc3 = self.decoder(128, 128, kernel_size=2, stride=2, bias=False)
        self.dc2 = self.decoder(64 + 128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc1 = self.decoder(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.dc0 = self.decoder(64, self.n_classes, kernel_size=1, stride=1, bias=False)

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
    
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=True, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.ReLU())
        return layer


    def decoder(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def forward(self, x):
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e0, e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)
        del e5, e6

        d9 = torch.cat((self.dc9(e7), syn2), dim=1)
        del e7, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), dim=1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), dim=1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        return d0

    def configure_optimizers(self):
        from torch.optim import AdamW
        # optim = Adam(self.parameters(), lr=1e-3)
        optim = AdamW(self.parameters(), lr=1e-3, eps=1e-3, weight_decay=1e-4)
        return optim

    def training_step(self, batch, batch_idx):
        data, nodule = batch
        out = self(data)
        # loss1 = self.criterion(out, nodule)
        loss2 = self.bce(out, nodule)
        # loss = loss1 * 4 + loss2
        with torch.no_grad():
            dice = self.dice_coefficient(out.cpu().detach(), nodule.cpu().detach())
            self.log_dict({"dice": dice}, prog_bar=True)
        return loss2

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
