from typing import Tuple
import torch
import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule
from .metric import DiceLoss, DiceCoefficient

class BasicBlock(nn.Module):

    def __init__(self, inplane, outplane, downsample=True):
        super(BasicBlock, self).__init__()
        stride = 2 if downsample else 1
        if downsample:
            self.skip_connection = nn.Conv3d(inplane, outplane, kernel_size=3, stride=2, padding=1)
        else:
            self.skip_connection = None
        self.conv1 = nn.Conv3d(inplane, outplane, kernel_size=3, stride=stride, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(outplane, outplane, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        if self.skip_connection:
            skip = self.skip_connection(x)
        else:
            skip = x
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)

        x += skip
        x = self.relu2(x)
        return x


class ResBlock(nn.Module):
    
    def __init__(self, inplane, outplane, num_layers, down_sample=True):
        super(ResBlock, self).__init__()

        self.block1 = BasicBlock(inplane=inplane, outplane=outplane, downsample=down_sample)

        layers = []
        for i in range(1, num_layers):
            layers.append(BasicBlock(inplane=outplane, outplane=outplane, downsample=False))
        
        self.other_blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.other_blocks(self.block1(x))


class ResUnet(LightningModule):

    def __init__(self, inplane, channels=[2, 4, 8, 16]):
        """
        down -> down -> down -> up
        (W, H, C) -> (W/4, H/4, C/4)
        """
        super(ResUnet, self).__init__()

        self.conv1 = ResBlock(inplane, channels[0], num_layers=2, down_sample=False)
        self.down_layer1 = nn.Conv3d(channels[0], channels[0], kernel_size=(3, 3, 3), stride=2, padding=1)

        self.conv2 = ResBlock(channels[0], channels[1], num_layers=2, down_sample=False)
        self.down_layer2 = nn.Conv3d(channels[1], channels[1], kernel_size=(3, 3, 3), stride=2, padding=1)

        self.conv3 = ResBlock(channels[1], channels[2], num_layers=2, down_sample=False)
        self.down_layer3 = nn.Conv3d(channels[2], channels[2], kernel_size=(3, 3, 3), stride=2, padding=1)

        self.conv = ResBlock(channels[2], channels[2], num_layers=2, down_sample=False)

        self.up_layer1 = nn.ConvTranspose3d(channels[2], channels[2], kernel_size=2, padding=0, stride=2)
        self.conv4 = ResBlock(channels[3], channels[2], num_layers=2, down_sample=False)

        self.criterion = DiceLoss()

        self.dice_coefficient = DiceCoefficient()

    def forward(self, x):
        output1 = self.conv1(x)
        x = self.down_layer1(output1)
        output2 = self.conv2(x)
        x = self.down_layer2(output2)
        output3 = self.conv3(x)
        x = self.down_layer3(output3)

        x = self.conv(x)

        x = self.up_layer1(x)
        x = self.conv4(torch.cat([output3, x], dim=1))
        return x

    def configure_optimizers(self):
        from torch.optim import SGD
        optimizer = SGD(self.parameters(), lr=1e-3, momentum=0.2, weight_decay=1e-4)
        return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        mhd, nodule = batch
        out = self(mhd)
        loss = self.criterion(out, nodule)
        self.log("loss", loss)
        return loss

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        mhd, nodule = batch
        out = self(mhd)
        dice_coff = self.dice_coefficient(out, nodule)
        self.log("dice cofficient", dice_coff)
        return dice_coff