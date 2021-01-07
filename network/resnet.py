from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.init import *
from torch.cuda.amp import autocast, GradScaler


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplane, outplane, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplane, outplane, (1, 1, 1), stride=1, bias=False)
        self.bn1 = nn.BatchNorm3d(outplane)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv3d(outplane, outplane, (3, 3, 3), stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(outplane)
        self.relu2 = nn.PReLU()

        self.conv3 = nn.Conv3d(outplane, outplane*self.expansion, (1, 1, 1), stride=1, bias=False)
        self.bn3 = nn.BatchNorm3d(outplane*self.expansion)
        self.relu3 = nn.PReLU()

        if stride == 2:
            self.shortcut = nn.Conv3d(inplane, outplane*self.expansion, (1, 1, 1), stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        res = x
        if self.shortcut:
            res = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += res
        x = self.relu3(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, inplane, outplane, stride=1, activateion=nn.ReLU):
        super(ResBlock, self).__init__()
        if stride == 2:
            self.shortcut = nn.Conv3d(inplane, outplane, 
                                      kernel_size=1,
                                      stride=stride)
        else:
            self.shortcut = None
        
        self.conv1 = nn.Conv3d(inplane, outplane, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(outplane)
        self.relu1 = activateion()

        self.conv2 = nn.Conv3d(outplane, outplane, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm3d(outplane)
        self.relu2 = activateion()

        self.init_weight()
        
    def forward(self, x):
        res = x
        if self.shortcut:
            res = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)

        x += res
        x = self.relu2(x)
        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                normal_(m.weight)


# class ResNet3D(nn.Module):
#     """
#     original network 1
#     """
#     def __init__(self, block_num=[5, 10]):
#         super(ResNet3D, self).__init__()
#         Block = Bottleneck

#         plane = 32
#         self.conv0 = nn.Sequential(
#             nn.Conv3d(1, plane, (3, 3, 3), padding=1),
#             nn.BatchNorm3d(plane),
#             nn.PReLU(),
#             nn.Conv3d(plane, plane, (3, 3, 3), padding=1),
#             nn.BatchNorm3d(plane),
#             nn.PReLU()   
#         )

#         self.conv1 = []
#         for i in range(block_num[0]):
#             if i == 0:
#                 self.conv1.append(Block(plane, 128, stride=2))
#             else:
#                 self.conv1.append(Block(128*Block.expansion, 128))
#         self.conv1 = nn.Sequential(*self.conv1)  # /2

#         plane = 128 * Block.expansion
#         self.conv2 = []
#         for i in range(block_num[1]):
#             if i == 0:
#                 self.conv2.append(Block(plane, 256, stride=2))
#             else:
#                 self.conv2.append(Block(256*Block.expansion, 256))
#         self.conv2 = nn.Sequential(*self.conv2)

#         plane = 256 * Block.expansion
#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, 1)),
#             nn.Flatten(start_dim=1),
#             nn.Dropout(),
#             nn.Linear(plane, plane*2),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(plane*2, plane//2),
#             nn.ReLU(),
#             nn.Dropout(),
#             nn.Linear(plane//2, 2)
#             # nn.Sigmoid(), 
#             # nn.Softmax(dim=1)
#         )
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x, enable_amp=True):
#         x = self.conv0(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         with autocast():
#             x = self.fc(x)
#         x = torch.sigmoid(x)
#         return x


class ResNet3D(nn.Module):
    """
    resnet_deeper_ppp: still to be ReLU
    """
    def __init__(self):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Sequential(
            ResBlock(1, 128, stride=2),
            ResBlock(128, 128)
            # nn.Dropout3d(),
            # nn.Conv3d(1, 12, stride=2, kernel_size=3, padding=1),
            # nn.BatchNorm3d(64),
            # nn.ReLU(),
        )  # /2

        self.conv2 = nn.Sequential(
            # nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            # nn.Dropout3d(),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 256),
        )  # /4

        self.conv3 = nn.Sequential(
            # nn.Dropout3d(),
            ResBlock(256, 512, stride=2),
            ResBlock(512, 512)
        )  # /8

        self.conv4 = nn.Sequential(
            # nn.Dropout3d(),
            ResBlock(512, 1024, stride=2),
            ResBlock(1024, 1024)
        )  # /16

        self.conv5 = nn.Sequential(
            # nn.Dropout3d(),
            ResBlock(1024, 2048, stride=2),
            ResBlock(2048, 2048)
        )  # /32

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(start_dim=1),
            # nn.Dropout(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(1024, 2),
            # nn.Sigmoid(), 
            # nn.Softmax(dim=1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


# class ResNet3D(nn.Module):
#     """
#     resnet_deeper_plus_plus
#     """
#     def __init__(self):
#         super(ResNet3D, self).__init__()
#         self.conv1 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(1, 64, stride=2),
#             ResBlock(64, 64)
#             # nn.Conv3d(1, 32, stride=2, kernel_size=3, padding=1),
#             # nn.BatchNorm3d(32),
#             # nn.ReLU(),
#         )  # /2

#         self.conv2 = nn.Sequential(
#             ResBlock(64, 128, stride=2),
#             ResBlock(128, 128),
#         )  # /4

#         self.conv3 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(128, 256, stride=2),
#             ResBlock(256, 256)
#         )  # /8

#         self.conv4 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(256, 512, stride=2),
#             ResBlock(512, 512)
#         )  # /16

#         self.conv5 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(512, 1024, stride=2),
#             ResBlock(1024, 1024)
#         )  # /32


#         self.fc = nn.Sequential(
#             # nn.AdaptiveAvgPool3d((1, 1, 1)),
#             nn.Flatten(start_dim=1),
#             # nn.Dropout(),
#             nn.Linear(1024, 1024),
#             nn.Sigmoid(),
#             # # nn.Dropout(),
#             # nn.Linear(2048, 2048),
#             # nn.ReLU(),
#             # nn.Dropout(),
#             nn.Linear(1024, 2),
#             # nn.Sigmoid(), 
#             # nn.Softmax(dim=1)
#         )
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.fc(x)
#         x = torch.sigmoid(x)
#         return x

# class ResNet3D(nn.Module):
#     """
#     resnet_deeper_ppp_1: still to be ReLU
#     """
#     def __init__(self):
#         super(ResNet3D, self).__init__()
#         self.conv1 = nn.Sequential(
#             ResBlock(1, 64, stride=2),
#             ResBlock(64, 64),
#             ResBlock(64, 64),
#             ResBlock(64, 64),
#         )  # /2

#         self.conv2 = nn.Sequential(
#             # nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
#             # nn.Dropout3d(),
#             ResBlock(64, 128, stride=2),
#             ResBlock(128, 128),
#             ResBlock(128, 128),
#             ResBlock(128, 128),
#         )  # /4

#         self.conv3 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(128, 256, stride=2),
#             ResBlock(256, 256),
#             ResBlock(256, 256),
#             ResBlock(256, 256),
#         )  # /8

#         self.conv4 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(256, 512, stride=2),
#             ResBlock(512, 512),
#             ResBlock(512, 512),
#             ResBlock(512, 512),
#         )  # /16

#         self.conv5 = nn.Sequential(
#             # nn.Dropout3d(),
#             ResBlock(512, 1024, stride=2),
#             ResBlock(1024, 1024),
#             ResBlock(1024, 1024),
#             ResBlock(1024, 1024),
#         )  # /32

#         self.fc = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, 1)),
#             nn.Flatten(start_dim=1),
#             nn.Dropout(),
#             nn.Linear(1024, 1024),
#             nn.PReLU(),
#             nn.Dropout(),
#             nn.Linear(1024, 512),
#             nn.PReLU(),
#             nn.Dropout(),
#             nn.Linear(512, 2),
#             # nn.Sigmoid(), 
#             # nn.Softmax(dim=1)
#         )
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x, enable_amp):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         with autocast(enable_amp):
#             x = self.fc(x)
#         x = torch.sigmoid(x)
#         return x

# if __name__ == "__main__":
#     net = ResNet3D()
#     print(net)