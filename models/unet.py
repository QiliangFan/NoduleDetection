from __future__ import print_function


import torch
import torch.nn as nn


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


def make_layer3d(inplane, plane, *,down_sample=False):
    if not down_sample:
        return nn.Sequential(
            make_conv3d(inplane, plane),
            make_conv3d(plane, plane)
        )
    else:
        return nn.Sequential(
            make_conv3d(inplane, plane, stride=2),
            make_conv3d(plane, plane)
        )


class Unet3D(nn.Module):
    def __init__(self, channels=[1, 10, 20, 40], num_classes=1):
        """
        :param conv_dim: If 3, then create 3D-Unet. If 2, then create 2D-Unet
        """
        super(Unet3D, self).__init__()
        # encoder
        self.down_layer1 = make_layer3d(channels[0], channels[1])
        self.down_layer2 = make_layer3d(channels[1], channels[2], down_sample=True)
        self.down_layer3 = make_layer3d(channels[2], channels[3], down_sample=True)
        self.down_layer4 = make_layer3d(channels[3], channels[3], down_sample=True)

        # decoder
        self.up_conv1 = make_transconv3d(channels[3], channels[3])
        self.up_layer1 = make_layer3d(channels[3]*2, channels[2])

        self.up_conv2 = make_transconv3d(channels[2], channels[2])
        self.up_layer2 = make_layer3d(channels[2]*2, channels[1])

        self.up_conv3 = make_transconv3d(channels[1], channels[1])
        self.up_layer3 = make_layer3d(channels[1]*2, num_classes)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output1 = self.down_layer1(x)
        output2 = self.down_layer2(output1)
        output3 = self.down_layer3(output2)
        feature = self.down_layer4(output3)
        
        x = self.up_conv1(feature)
        x = self.up_layer1(torch.cat([x, output3], dim=1))
        x = self.up_conv2(x)
        x = self.up_layer2(torch.cat([x, output2], dim=1))
        x = self.up_conv3(x)
        x = self.up_layer3(torch.cat([x, output1], dim=1))
        x = self.sigmoid(x)
        del output1, output2, output3
        return x