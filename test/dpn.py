from __future__ import print_function
from typing import Tuple, List, Any
from collections import OrderedDict
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch.nn.modules import padding
from torch.nn import BCELoss
from torch.nn.modules.module import T
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

    def __init__(self, inc: int, in_feature: int, out_1x1_ch1: int, out_3x3_ch: int, out_1x1_ch2: int, groups: int = 1, down_sample: bool = False, project: bool = False):
        """
        inc: increment of Dense network
        """
        super().__init__()
        self.inc = inc
        self.in_feature = in_feature
        self.out_1x1_ch1 = out_1x1_ch1
        self.out_3x3_ch = out_3x3_ch
        self.out_1x1_ch2 = out_1x1_ch2
        self.down_sample = down_sample

        if project:
            self.project = nn.Conv3d(in_feature, out_1x1_ch2 + 2*inc, kernel_size=1, stride=(2 if down_sample else 1))
        else:
            self.project = None

        self.conv1 = nn.Conv3d(
            (out_1x1_ch2 + 2*inc if down_sample else in_feature), out_1x1_ch1, (1, 1, 1), padding=0)
        self.bn1 = nn.BatchNorm3d(out_1x1_ch1, eps=1e-3)
        self.relu1 = nn.ReLU(inplace=True)

        assert out_3x3_ch % groups == 0
        self.conv2 = nn.Conv3d(out_1x1_ch1, out_3x3_ch,
                               (3, 3, 3), padding=1, stride=1, groups=groups)
        self.bn2 = nn.BatchNorm3d(out_3x3_ch, eps=1e-3)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(out_3x3_ch, out_1x1_ch2+inc,
                               (1, 1, 1), padding=0, stride=1)
        self.bn3 = nn.BatchNorm3d(out_1x1_ch2+inc, eps=1e-3)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        data_in = torch.cat(x, dim=1) if isinstance(x, Tuple) else x
        if self.project is not None:
            data_x = self.project(data_in)
            if self.down_sample:
                data_in = data_x
            res_x, dense_x = data_x[:, :self.out_1x1_ch2], data_x[:, self.out_1x1_ch2:]
        else:
            res_x, dense_x = x

        data_in = self.conv1(data_in)
        data_in = self.bn1(data_in)
        data_in = self.relu1(data_in)

        data_in = self.conv2(data_in)
        data_in = self.bn2(data_in)
        data_in = self.relu2(data_in)

        data_in = self.conv3(data_in)
        data_in = self.bn3(data_in)
        data_in = self.relu3(data_in)

        res_out = data_in[:, :self.out_1x1_ch2, :, :, :]
        dense_out = data_in[:, self.out_1x1_ch2:, :, :, :]
        dense_out = torch.cat([dense_x, dense_out], dim=1)
        res_out = res_x + res_out
        return res_out, dense_out


class DPN(LightningModule):

    def __init__(self, init_feature: int, groups: int, blocks: List[int] = [3, 4, 4, 3], channels: List[int] = [96, 192, 384, 768], inc: List = [16, 32, 24, 128], save_dir: str = None):
        super().__init__()
        self.save_dir = save_dir

        self.input_block = InputBlock(1, init_feature)
        out_1x1_channels = [256, 512, 1024, 2048]

        layers = OrderedDict()

        layers["layer0_0"] = DualPathBlock(inc[0], init_feature, out_1x1_ch1=channels[0], out_3x3_ch=channels[0], out_1x1_ch2=out_1x1_channels[0], groups=groups, project=True)
        in_chs = out_1x1_channels[0] + 3 * inc[0]
        for i in range(1, blocks[0]):
            layers[f"layer0_{i}"] = DualPathBlock(inc[0], in_chs, out_1x1_ch1=channels[0], out_3x3_ch=channels[0], out_1x1_ch2=out_1x1_channels[0], groups=groups)
            in_chs += inc[0]

        layers["layer1_0"] = DualPathBlock(inc[1], in_chs, out_1x1_ch1=channels[1], out_3x3_ch=channels[1], out_1x1_ch2=out_1x1_channels[1], groups=groups, down_sample=True, project=True)
        in_chs = out_1x1_channels[1] + 3 * inc[1]
        for i in range(1, blocks[1]):
            layers[f"layer2_{i}"] = DualPathBlock(inc[1], in_chs, out_1x1_ch1=channels[1], out_3x3_ch=channels[1], out_1x1_ch2=out_1x1_channels[1], groups=groups)
            in_chs += inc[1]

        layers["layer2_0"] = DualPathBlock(inc[2], in_chs, out_1x1_ch1=channels[2], out_3x3_ch=channels[2], out_1x1_ch2=out_1x1_channels[2], groups=groups, down_sample=True, project=True)
        in_chs = out_1x1_channels[2] + 3 * inc[2]
        for i in range(1, blocks[2]):
            layers[f"layer2_{i}"] = DualPathBlock(inc[2], in_chs, out_1x1_ch1=channels[2], out_3x3_ch=channels[2], out_1x1_ch2=out_1x1_channels[2], groups=groups)
            in_chs += inc[2]
        
        layers["layer3_0"] = DualPathBlock(inc[3], in_chs, out_1x1_ch1=channels[3], out_3x3_ch=channels[3], out_1x1_ch2=out_1x1_channels[3], groups=groups, down_sample=True, project=True)
        in_chs = out_1x1_channels[3] + 3 * inc[3]
        for i in range(1, blocks[3]):
            layers[f"layer3_{i}"] = DualPathBlock(inc[3], in_chs, out_1x1_ch1=channels[3], out_3x3_ch=channels[3], out_1x1_ch2=out_1x1_channels[3], groups=groups)
            in_chs += inc[3]

        self.feature = nn.Sequential(layers)

        self.classify = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(out_1x1_channels[3], 1, kernel_size=1, padding=0, stride=1),
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
        x = self.input_block(x)

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
            self.log("acc", self.acc_meter.avg, prog_bar=True)
        self.log("loss", loss)
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