import torch
import numpy as np
import torch.nn as nn
from pytorch_lightning import LightningModule


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
            pass


class ResBlock(nn.Module):
    
    def __init__(self, inplane, outplane, num_layers, down_sample=True):
        super(ResBlock, self).__init__()


    def forward(self, x):
        pass


class ResUnet(LightningModule):

    def __init__(self):
        super(ResUnet, self).__init__()

    def forward(self, x):
        pass
