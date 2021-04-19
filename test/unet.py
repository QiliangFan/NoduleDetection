from __future__ import print_function
from network.metrics import AccMeter, AverageMeter

from typing import Tuple
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


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


class Unet3D(LightningModule):
    def __init__(self, channels=[1, 2, 4, 8, 16, 32], save_dir: str = None):
        """
        :param conv_dim: If 3, then create 3D-Unet. If 2, then create 2D-Unet
        """
        super(Unet3D, self).__init__()
        self.save_dir = save_dir

        # encoder
        self.down_layer1 = make_layer3d(channels[0], channels[1])
        self.down_layer2 = make_layer3d(channels[1], channels[2], down_sample=True)
        self.down_layer3 = make_layer3d(channels[2], channels[3], down_sample=True)
        self.down_layer4 = make_layer3d(channels[3], channels[4], down_sample=True)
        self.down_layer5 = make_layer3d(channels[4], channels[4], down_sample=True)

        # decoder
        self.up_conv1 = make_transconv3d(channels[4], channels[4])
        self.up_layer1 = make_layer3d(channels[4]*2, channels[3])

        self.up_conv2 = make_transconv3d(channels[3], channels[3])
        self.up_layer2 = make_layer3d(channels[3]*2, channels[2])

        self.up_conv3 = make_transconv3d(channels[2], channels[2])
        self.up_layer3 = make_layer3d(channels[2]*2, channels[1])

        self.up_conv4 = make_transconv3d(channels[1], channels[1])
        self.up_layer4 = make_layer3d(channels[1]*2, channels[0])  # dice loss
        # self.up_layer5 = make_layer3d(channels[1]*2, 2)  # cross-entropy

        self.reduce = nn.Sequential(
            nn.AdaptiveMaxPool3d((4, 4, 4)),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

        self.sigmoid = nn.Sigmoid()
                # criterion
        from torch.nn import BCELoss
        self.bce_loss = BCELoss()
        self.acc = AccMeter()

        self.tp_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fn_meter = AverageMeter()

    def forward(self, x):
        output1 = self.down_layer1(x)
        output2 = self.down_layer2(output1)
        output3 = self.down_layer3(output2)
        output4 = self.down_layer4(output3)
        feature = self.down_layer5(output4)
        
        x = self.up_conv1(feature)
        x = self.up_layer1(torch.cat([x, output4], dim=1))
        x = self.up_conv2(x)
        x = self.up_layer2(torch.cat([x, output3], dim=1))
        x = self.up_conv3(x)
        x = self.up_layer3(torch.cat([x, output2], dim=1))
        x = self.up_conv4(x)
        x = self.up_layer4(torch.cat([x, output1], dim=1))
        x = self.reduce(x)
        x = self.sigmoid(x)
        x = x.view(x.shape[0], -1)
        del output1, output2, output3, output4
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x

    def training_step(self, batch, batch_idx):
        ct, nodule = batch
        out = self(ct)
        loss = self.bce_loss(out, nodule)
        with torch.no_grad():
            self.acc.update(out.detach().cpu(), nodule.detach().cpu())
        self.log("acc", self.acc.avg, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.acc.reset()

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        ct, nodule = batch
        out = self(ct)

        if self.save_dir is not None:
            import os
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            with open(os.path.join(self.save_dir, "output.txt"), "a") as fp:
                for _out, _nodule in zip(out.cpu(), nodule.cpu()):
                    if _out.item() > 0.5 and _nodule == 1:  # TP
                        self.tp_meter.update(1, 1)
                        self.fp_meter.update(0, 1)
                        self.tn_meter.update(0, 1)
                        self.fn_meter.update(0, 1)
                    elif _out.item() > 0.5 and _nodule == 0:  # FP
                        self.tp_meter.update(0, 1)
                        self.fp_meter.update(1, 1)
                        self.tn_meter.update(0, 1)
                        self.fn_meter.update(0, 1)
                    elif _out.item() <= 0.5 and _nodule == 1:  # FN
                        self.tp_meter.update(0, 1)
                        self.fp_meter.update(0, 1)
                        self.tn_meter.update(0, 1)
                        self.fn_meter.update(1, 1)
                    else:  # TN
                        self.tp_meter.update(0, 1)
                        self.fp_meter.update(0, 1)
                        self.tn_meter.update(1, 1)
                        self.fn_meter.update(0, 1)
                    print(_out.item(), _nodule.item(), sep=",", file=fp)
            self.log_dict({
                "tp": self.tp_meter.total,
                "fp": self.fp_meter.total,
                "tn": self.tn_meter.total,
                "fn": self.fn_meter.total,
            }, prog_bar=True)
        self.acc.update(out.detach().cpu(), nodule.detach().cpu())
        self.log("acc", self.acc.avg, prog_bar=True)
        return batch_idx

    def test_epoch_end(self, outputs):
        tp = self.tp_meter.total
        tn = self.tn_meter.total
        fp = self.fp_meter.total
        fn = self.fn_meter.total

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        self.log_dict({"precision": precision, "recall": recall}, prog_bar=True)

        with open("metrics.txt", "a") as fp:
            import json
            result = self.trainer.logged_metrics
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].item()
            result = json.dumps(result, indent=4)
            print(result, file=fp)
        
        self.acc.reset()
        self.tp_meter.reset()
        self.tn_meter.reset()
        self.fp_meter.reset()
        self.fn_meter.reset()

    def configure_optimizers(self):
        from torch.optim import SGD

        sgd = SGD(self.parameters(), lr=1e-3, momentum=0.1, weight_decay=1e-4)
        return sgd
