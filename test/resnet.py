from typing import Sequence
import torch
import torch.nn as nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.dropout import Dropout
from torch.utils.data import DataLoader


def conv333(inchannel, outchannel, stride=1):
    return nn.Conv3d(inchannel,
                     outchannel,
                     (3, 3, 3),
                     stride=stride,
                     padding=1)
                

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv333(inchannel,
                             outchannel,
                             stride=stride)
        self.bn1 = nn.BatchNorm3d(outchannel)
        
        self.conv2 = conv333(inchannel,
                             outchannel,
                             stride=1)
        self.bn2 = nn.BatchNorm3d(outchannel)

        if stride == 2:
            self.short_cut = nn.Conv3d(inchannel,
                                       outchannel,
                                       (1, 1, 1),
                                       stride=stride)
        else:
            self.short_cut = None

    def forward(self, x):
        res = x
        if self.short_cut:
            res = self.short_cut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = self.relu(x)
        return x
                        

class Resnet3D(nn.Module):
    def __init__(self, in_channel, num_classes, verbose=False):
        super(Resnet3D, self).__init__()
        self.verbose = verbose
        # 1*24*40*40
        self.block1 = nn.Conv3d(in_channel,
                                64, 
                                (1, 1, 1),
                                stride=1)  # 64*24*40*40
        self.block2 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),  # 64*24*20*20
            ResBlock(64, 64),
            ResBlock(64, 64), # 64*24*20*20
            nn.Dropout()
        )

        self.block3 = nn.Sequential(
            ResBlock(64, 128, 2), # 128*12*10*10
            ResBlock(128, 128),
            nn.Dropout()
        )

        self.block4 = nn.Sequential(
            ResBlock(128, 256, 2), # 256*6*5*5
            nn.Conv3d(256, 256, (1, 2, 2), stride=0),  # 256*6*4*4
            ResBlock(256, 256),
            nn.Dropout()
        )

        self.blcok5 = nn.Sequential(
            ResBlock(256, 512, 2), # 512*3*2*2
            nn.Conv3d(512, 512, (2, 1, 1), stride=1),  # 512*2*2*2
            ResBlock(512, 512),
            nn.AvgPool3d((2, 2, 2), 1), # 512*1*1*1
            nn.Dropout()
        )

        self.fc = Sequential(
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        if self.verbose:
            print(f"block1 output shape: {x.shape}")
        x = self.block2(x)
        if self.verbose:
            print(f"block2 output shape: {x.shape}")
        x = self.block3(x)
        if self.verbose:
            print(f"block3 output shape: {x.shape}")
        x = self.block4(x)
        if self.verbose:
            print(f"block4 output shape: {x.shape}")
        x = self.block5(x)
        if self.verbose:
            print(f"block5 output shape: {x.shape}")
        x = self.fc(x)
        if self.verbose:
            print(f"fc layer output shape: {x.shape} - {x.data}")
        return x
        
        