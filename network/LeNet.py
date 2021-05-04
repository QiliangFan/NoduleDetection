"""
LeNet-5
"""
import os
import json
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.nn.modules.batchnorm import BatchNorm3d
from network.metrics import AverageMeter, AccMeter
from collections import OrderedDict
from typing import Tuple, List

class LeNet(LightningModule):

    def __init__(self, shape: Tuple, channels=[1, 6, 6, 16], save_dir: str = None):
        super(LeNet, self).__init__()
        assert len(shape) == 3 and len(set(shape)) == 1

        self.save_dir = save_dir
        self.shape_input = shape[0]
        cur_shape = self.shape_input
        self.init_meter()

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=channels[0], out_channels=channels[1], kernel_size=5, stride=1, padding=0),
            nn.AvgPool3d(kernel_size=2, stride=2),
            BatchNorm3d(channels[1], eps=1e-3),
            nn.ReLU(inplace=True)
        )
        cur_shape = (cur_shape - 4) // 2

        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=channels[1], out_channels=channels[2], kernel_size=5, stride=1, padding=0),
            nn.AvgPool3d(kernel_size=2, stride=2),
            BatchNorm3d(channels[2], eps=1e-3),
            nn.ReLU(inplace=True),
        )
        cur_shape = (cur_shape - 4) // 2

        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=channels[2], out_channels=channels[3], kernel_size=5, stride=1, padding=0),
            nn.AvgPool3d(kernel_size=2, stride=2),
            BatchNorm3d(channels[3], eps=1e-3),
            nn.ReLU(inplace=True),

        )
        cur_shape = (cur_shape - 4) // 2

        self.classify = nn.Sequential(OrderedDict([
            ("flatten_layer", nn.Flatten()),
            ('fc1', nn.Linear(channels[3]*cur_shape**3, 120)),
            ("'relu1", nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(120, 32)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc3', nn.Linear(32, 1)),
            ('sigmoid', nn.Sigmoid())
        ]))

        # criterion
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classify(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        out = self(data)
        loss = self.criterion(out, target)
        return loss

    def training_epoch_end(self, outputs: List):
        self.reset_meter()

    def validation_step(self, batch, batch_idx):
        self.inference(batch, batch_idx)

    def validation_epoch_end(self, outputs: List) -> None:
        self.reset_meter()

    def test_step(self, batch, batch_idx):
        self.inference(batch, batch_idx, save=True)

    def test_epoch_end(self, outpus: List) -> None:
        if self.save_dir:
            metrics = self.trainer.logged_metrics
            for key in metrics:
                if isinstance(metrics[key], torch.Tensor):
                    metrics[key] = metrics[key].item()
            with open(os.path.join(self.save_dir, "metrics.txt"), "a") as fp:
                print(json.dumps(metrics, indent=4), file=fp)

        self.reset_meter()

    def configure_optimizers(self):
        from torch.optim import Adam

        optim = Adam(self.parameters(), lr=1e-3)
        return optim

    @torch.no_grad()
    def inference(self, batch, batch_idx, save=False):
        data, target = batch
        out = self(data)
        precision, recall = self.precision_recall(out, target)
        if save and self.save_dir is not None:
            with open(os.path.join(self.save_dir, "output.txt"), "a") as fp:
                for _out, _target in zip(out.cpu(), target.cpu()):
                    print(_out, _target, sep=",", file=fp)

        self.log_dict({
            "precision": precision,
            "recall": recall,
            "tp": self.tp_meter.total,
            "tn": self.tn_meter.total,
            "fp": self.fp_meter.total,
            "fn": self.fn_meter.total,
            "accuracy": self.acc_meter.avg
        }, prog_bar=True)
        return batch_idx

    def init_meter(self):
        self.tp_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fn_meter = AverageMeter()
        self.acc_meter = AccMeter()

    def reset_meter(self):
        self.tp_meter.reset()
        self.fp_meter.reset()
        self.tn_meter.reset()
        self.fn_meter.reset()
        self.acc_meter.reset()

    def precision_recall(self, out: torch.Tensor, target: torch.Tensor):
        out, target = out.cpu().numpy(), target.cpu().numpy()
        out[out > 0.5] = 1
        out[out <= 0.5] = 0

        # TP
        tmp = out * target
        tp_num = int(tmp.sum())

        # TN
        tmp = 1 - out
        tmp_target = 1 - target
        tmp = tmp * tmp_target
        tn_num = int(tmp.sum())

        # FP 
        tmp_target = 1 - target
        tmp = out * tmp_target
        fp_num = int(tmp.sum())

        # FN
        tmp = 1 - out
        tmp = tmp * target
        fn_num = int(tmp.sum())
        
        # ACC
        acc = tp_num + tn_num
        total = tp_num + tn_num + fp_num + fn_num

        self.tp_meter.update(tp_num)
        self.tn_meter.update(tn_num)
        self.fp_meter.update(fp_num)
        self.fn_meter.update(fn_num)
        self.acc_meter.update(out, target)

        tp = self.tp_meter.total
        tn = self.tn_meter.total
        fp = self.fp_meter.total
        fn = self.fn_meter.total
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        return precision, recall