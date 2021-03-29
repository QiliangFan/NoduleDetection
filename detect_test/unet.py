from __future__ import print_function

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Sequential
from torch.optim import Adam

from metrics import DiceLoss


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

    def __init__(self, channels=[1, 2, 4, 8, 16, 32]):
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

        self.layer4 = nn.Sequential(
            make_conv3d(channels[4], channels[5], stride=2),  # 16
            make_conv3d(channels[5], channels[5])
        )

        self.up_conv0 = make_transconv3d(channels[5], channels[4])
        self.up_layer0 = nn.Sequential(
            make_conv3d(channels[5], channels[4]),
            make_conv3d(channels[4], channels[4])
        )

        self.up_conv1 = make_transconv3d(channels[4], channels[3])
        self.up_layer1 = nn.Sequential(
            make_conv3d(channels[4], channels[3]),
            make_conv3d(channels[3], channels[3])
        )

        self.up_conv2 = make_transconv3d(channels[3], channels[2])
        self.up_layer2 = nn.Sequential(
            make_conv3d(channels[3], channels[2]),
            make_conv3d(channels[2], channels[2])
        )

        self.up_conv3 = make_transconv3d(channels[2], channels[1])
        self.up_layer3 = nn.Sequential(
            make_conv3d(channels[2], channels[1]),
            make_conv3d(channels[1], channels[1])
        )

        self.conv = make_conv3d(channels[1], channels[0])
        self.activation = nn.Sigmoid()

        self.criterion = DiceLoss()

        # output file
        self.fp = open("output.txt", "w")

    def forward(self, x) -> Tensor:
        output0 = self.layer0(x)
        output1 = self.layer1(output0)
        output2 = self.layer2(output1)
        output3 = self.layer3(output2)
        output4 = self.layer4(output3)

        output = self.up_conv0(output4)
        output = self.up_layer0(torch.cat([output3, output], dim=1))

        output = self.up_conv1(output)
        output = self.up_layer1(torch.cat([output2, output], dim=1))

        output = self.up_conv2(output)
        output = self.up_layer2(torch.cat([output1, output], dim=1))

        output = self.up_conv3(output)
        output = self.up_layer3(torch.cat([output0, output], dim=1))
        output = self.conv(output)
        output = self.activation(output)
        return output

    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-4)
        return optim

    def training_step(self, batch, batch_idx):
        data, nodule = batch
        out = self(data)
        loss = self.criterion(out, nodule)
        self.log_dict({"loss": loss.item()})
        return loss

    def test_step(self, batch, batch_idx):
        """
        阈值: 0.8
        """
        threshold = 0.8
        data, nodule = batch
        out: Tensor = self(data)
        out[out < threshold] = 0
        self.log_dict({"batch_idx": batch_idx}, prog_bar=True, on_step=True)

        args = out.argsort(descending=True)
        for bat, bat_data in enumerate(args):
            for ch, ch_data in enumerate(bat_data):
                for z, z_data in enumerate(ch_data):
                    for y, y_data in enumerate(z_data):
                        for x, data in enumerate(y_data):
                            if out[bat, ch, z, y, x] > 0:
                                min_z, max_z = max(
                                    0, z-5), min(args.shape[2]-1, z+5)
                                min_y, max_y = max(
                                    0, y-5), min(args.shape[3]-1, y+5)
                                min_x, max_x = max(
                                    0, x-5), min(args.shape[3]-1, x+5)
                                if torch.max(nodule[bat, ch, min_z:max_z, min_y:max_y, min_x:max_x]) > 0:
                                    label = 1
                                else:
                                    label = 0
                                print(out[bat, ch, z, y, x].item(),
                                      label, sep=",", file=self.fp)
                                out[bat, ch, min_z:max_z, min_y:max_y, min_x:max_x] = 0
        return batch_idx
