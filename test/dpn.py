from __future__ import print_function
from typing import Tuple, List, Any

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.nn.modules import padding
from torch.nn import BCELoss
from network.metrics import AccMeter, AverageMeter

class InputBlock(nn.Module):

    def __init__(self, in_feature: int, init_feature: int):
        super().__init__()

        self.conv1 = nn.Conv3d(in_feature, init_feature,
                               kernel_size=7, padding=3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm3d(init_feature, eps=1e-3)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):

    def __init__(self, inc: int, in_feature: int, out_1x1_ch1: int, out_3x3_ch: int, out_1x1_ch2: int, groups: int = 1, down_sample: bool = False):
        """
        inc: increment of Dense network
        """
        super().__init__()
        self.inc = inc
        self.in_feature = in_feature
        self.out_1x1_ch1 = out_1x1_ch1
        self.out_3x3_ch = out_3x3_ch
        self.out_1x1_ch2 = out_1x1_ch2

        self.conv1 = nn.Conv3d(
            in_feature, out_1x1_ch1, (1, 1, 1), padding=0, stride=(2 if down_sample else 1))
        self.bn1 = nn.BatchNorm3d(out_1x1_ch1, eps=1e-3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_1x1_ch1, out_3x3_ch,
                               (3, 3, 3), padding=1, stride=1, groups=groups)
        self.bn2 = nn.BatchNorm3d(out_3x3_ch, eps=1e-3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(out_3x3_ch, out_1x1_ch2,
                               (1, 1, 1), padding=0, stride=1)
        self.bn3 = nn.BatchNorm3d(out_1x1_ch2, eps=1e-3)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, Tuple) else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.relu3(x)
        dense_out = x[:, :self.inc, :, :, :]
        res_out = x[:, self.inc:, :, :, :]
        return dense_out, res_out


class DPN(LightningModule):

    def __init__(self, init_feature: int, groups: int, blocks: List[int] = [4, 4, 5, 4], channels: List[int] = [32, 64, 128, 256], inc: List = [16, 32, 32, 128], save_dir: str = None):
        super().__init__()
        self.save_dir = save_dir

        self.input_block = InputBlock(1, init_feature)

        block1 = [DualPathBlock(
            inc[0], init_feature, out_1x1_ch1=channels[0], out_3x3_ch=channels[0], out_1x1_ch2=256)]
        block1.extend([DualPathBlock(inc[0], 256, out_1x1_ch1=channels[0],
                                     out_3x3_ch=channels[0], out_1x1_ch2=256) for i in range(blocks[0]-1)])

        block2 = [DualPathBlock(inc[1], 256, out_1x1_ch1=channels[1],
                                out_3x3_ch=channels[1], out_1x1_ch2=512, down_sample=True)]
        block2.extend([DualPathBlock(inc[1], 512, out_1x1_ch1=channels[1],
                                     out_3x3_ch=channels[1], out_1x1_ch2=512) for i in range(blocks[1]-1)])

        block3 = [DualPathBlock(inc[2], 512, out_1x1_ch1=channels[2],
                                out_3x3_ch=channels[2], out_1x1_ch2=1024, down_sample=True)]
        block3.extend([DualPathBlock(
            inc[2], 1024, out_1x1_ch1=channels[2], out_3x3_ch=channels[2], out_1x1_ch2=1024) for i in range(blocks[2]-1)])

        block4 = [DualPathBlock(inc[3], 1024, out_1x1_ch1=channels[3],
                                out_3x3_ch=channels[3], out_1x1_ch2=2048, down_sample=True)]
        block4.extend([DualPathBlock(inc[3], 2048, out_1x1_ch1=channels[3],
                                     out_3x3_ch=channels[3], out_1x1_ch2=2048) for i in range(blocks[3]-1)])

        self.feature = nn.Sequential(*block1, *block2, *block3, *block4)

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(2048, 1, kernel_size=1, padding=0, stride=1),
            nn.Sigmoid()
        )

        # criterion
        self.loss = BCELoss()
        self.acc_meter = AccMeter()
        self.tp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.fn_meter = AverageMeter()

    def forward(self, x):
        x = self.feature(x)

        x = torch.cat(x, dim=1) if isinstance(x, Tuple) else x

        x = self.classify(x)

        x = x.view(x.shape[0], -1)
        return x

    def configure_optimizers(self):
        from torch.optim import SGD
        sgd = SGD(self.parameters(), lr=1e-3, momentum=0.1, weight_decay=1e-4)
        return sgd

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        data, target = batch
        out = self(data)
        loss = self.loss(out, target)

        with torch.no_grad():
            self.acc_meter.update(out.cpu(), target.cpu())
        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.acc_meter.reset()

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        data, target = batch
        out = self(data)
        
        # criterion
        self.acc_meter.update(out, target)
        if self.save_dir is not None:
            import os
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            with open(os.path.join(self.save_dir, "output.txt"), "a") as fp:
                for _out, _nodule in zip(out.cpu(), target.cpu()):
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
            self.acc_meter.update(out, target)
            self.log_dict({
                "tp": self.tp_meter.total,
                "fp": self.fp_meter.total,
                "tn": self.tn_meter.total,
                "fn": self.fn_meter.total,
                "acc": self.acc_meter.avg
            }, prog_bar=True)
        return batch_idx

    def test_epoch_end(self, outputs: List[Any]) -> None:
        tp = self.tp_meter.total
        tn = self.tn_meter.total
        fp = self.fp_meter.total
        fn = self.fn_meter.total

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        self.log_dict({"precision": precision, "recall": recall}, prog_bar=True)

        with open("test_dpn_metrics.txt", "a") as fp:
            import json
            result = self.trainer.logged_metrics
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].item()
            result = json.dumps(result, indent=4)
            print(result, file=fp)

        self.acc_meter.reset()
        self.tp_meter.reset()
        self.tn_meter.reset()
        self.fp_meter.reset()
        self.fn_meter.reset()