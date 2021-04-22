from torch.functional import Tensor
from network.meter import AverageMeter
from typing import Sequence
import torch
import os
import torch.nn as nn
from torch.nn.modules.container import Sequential
from torch.nn.modules.dropout import Dropout
from torch.utils.data import DataLoader
import torch.optim as optim
from pytorch_lightning import LightningModule
from typing import List, Any
from torchvision.utils import make_grid


def conv333(inchannel, outchannel, stride=1):
    return nn.Conv3d(inchannel,
                     outchannel,
                     (3, 3, 3),
                     stride=stride,
                     padding=1)
                

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.LeakyReLU(inplace=True)

        self.conv1 = conv333(inchannel,
                             outchannel,
                             stride=stride)
        self.bn1 = nn.BatchNorm3d(outchannel)
        
        self.conv2 = conv333(outchannel,
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
                        

class Resnet3D(LightningModule):
    def __init__(self, in_channel, num_classes, verbose=False, dropout=0.5, save_root=None):
        super(Resnet3D, self).__init__()
        self.save_root = save_root
        self.verbose = verbose
        # 1*24*40*40
        self.block1 = nn.Conv3d(in_channel,
                                16, 
                                (1, 1, 1),
                                stride=1)  # 16*48*48*48
        self.block2 = nn.Sequential(
            ResBlock(16, 32, 2),  # 16*24
            nn.Dropout3d(p=dropout),
            ResBlock(32, 32), 
            nn.Dropout3d(p=dropout),
            # nn.MaxPool3d((2, 2, 2), stride=2),  # 16*12
        )
        self.block3 = nn.Sequential(
            ResBlock(32, 64, stride=2), # 32*12
            nn.Dropout3d(p=dropout),
            ResBlock(64, 64),
            nn.Dropout3d(p=dropout),
            # nn.MaxPool3d((2, 2, 2), stride=2),  # 32*6
        )
        self.block4 = nn.Sequential(
            ResBlock(64, 128, stride=2), # 64*6
            nn.Dropout3d(p=dropout),
            ResBlock(128, 128),
            nn.Dropout3d(p=dropout),
        )
        self.block5 = nn.Sequential(
            ResBlock(128, 256, stride=2), # 128*3
            nn.Dropout(p=dropout),
            ResBlock(256, 256),
            nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool3d((3, 1, 1))
        )

        self.fc = Sequential(
            nn.Flatten(),
            nn.Linear(256*3, 256),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        # criterion and metric
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2]))
        self.acc_meter = AverageMeter()
        self.tp_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fn_meter = AverageMeter()

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            if self.verbose:
                self.logger.experiment.add_images(f"0input", x[0][0].unsqueeze(dim=1), self.batch_idx%1000)
        x = self.block1(x)
        with torch.no_grad():
            if self.verbose:
                self.logger.experiment.add_images(f"1block1", x[0][0].unsqueeze(dim=1), self.batch_idx%1000)
        x = self.block2(x)
        with torch.no_grad():
            if self.verbose:
                self.logger.experiment.add_images(f"2block2", x[0][0].unsqueeze(dim=1), self.batch_idx%1000)
        x = self.block3(x)
        with torch.no_grad():
            if self.verbose:
                self.logger.experiment.add_images(f"3block3", x[0][0].unsqueeze(dim=1), self.batch_idx%1000)
        x = self.block4(x)
        with torch.no_grad():
            if self.verbose:
                self.logger.experiment.add_images(f"4block4", x[0][0].unsqueeze(dim=1), self.batch_idx%1000)
        x = self.block5(x)
        x = self.fc(x)
        return x
        
    def training_step(self, batch: torch.Tensor, batch_idx):
        self.batch_idx = batch_idx
        data, target = batch
        out = self(data)
        loss = self.criterion(out, target)
        if self.verbose:
            self.precision_recall(out, target)
        return loss

    def training_step_end(self, output):
        if self.verbose:
            self.log_dict({
                "tp": self.tp_meter.sum,
                "fp": self.fp_meter.sum,
                "tn": self.tn_meter.sum,
                "fn": self.fn_meter.sum
            }, prog_bar=True, on_epoch=False, on_step=True)
        return output

    def training_epoch_end(self, outputs: List[Any]) -> None:
        if self.verbose:
            precision = self.tp_meter.sum / (self.tp_meter.sum + self.fp_meter.sum + 1e-6)
            recall = self.tp_meter.sum / (self.tp_meter.sum + self.fn_meter.sum + 1e-6)
            self.log_dict({
                "precision": precision,
                "recall": recall,
                "accuracy": self.acc_meter.avg
            }, prog_bar=True)
        self.acc_meter.reset()
        self.tp_meter.reset()
        self.fp_meter.reset()
        self.tn_meter.reset()
        self.fn_meter.reset()

    def test_step(self, batch, batch_idx):
        self.batch_idx = batch_idx
        data, target = batch
        out: Tensor = self(data)
        self.precision_recall(out, target)
        with open(os.path.join(self.save_root, "output.csv") if self.save_root else "output.csv", "a") as fp:
            for v, l in zip(out.cpu().squeeze().tolist(), target.cpu().squeeze().tolist()):
                print(v, l, sep=",", file=fp)

        self.log_dict({
            "tp": self.tp_meter.sum,
            "fp": self.fp_meter.sum,
            "tn": self.tn_meter.sum,
            "fn": self.fn_meter.sum
        }, on_epoch=False, prog_bar=True, on_step=True)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        precision = self.tp_meter.sum / (self.tp_meter.sum + self.fp_meter.sum + 1e-6)
        recall = self.tp_meter.sum / (self.tp_meter.sum + self.fn_meter.sum + 1e-6)
        self.log_dict({
            "precision": precision,
            "recall": recall,
            "accuracy": self.acc_meter.avg,
            "tp": self.tp_meter.sum,
            "fp": self.fp_meter.sum,
            "tn": self.tn_meter.sum,
            "fn": self.fn_meter.sum
        }, prog_bar=True)
        self.acc_meter.reset()
        self.tp_meter.reset()
        self.fp_meter.reset()
        self.tn_meter.reset()
        self.fn_meter.reset()

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), 
                        lr=1e-3, 
                        # momentum=0.2, 
                        # weight_decay=1e-4
                        )
        # stepLr = optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
        # return opt
        return {
            "optimizer": opt,
            # "lr_scheduler": stepLr
        }

    @torch.no_grad()
    def precision_recall(self, output: torch.Tensor, target: torch.Tensor):
        if hasattr(output, "cpu") and hasattr(target, "cpu"):
            output, target = output.cpu(), target.cpu()
        # tp
        tmp_output = output > 0.5
        tps = (tmp_output * target).sum()

        # fp
        tmp_target = target < 1
        fps = (tmp_output * tmp_target).sum()

        # tn
        tmp_output = output <= 0.5
        tmp_target = target < 1
        tns = (tmp_output * tmp_target).sum()

        # fn
        tmp_output = output <= 0.5
        fns = (tmp_output * target).sum()

        # acc
        tmp_output = output > 0.5
        acc_pos = (tmp_output * target).sum()
        tmp_output = output <= 0.5
        tmp_target = target < 1
        acc_neg = (tmp_output * tmp_target).sum()
        totals = target.numel()
        self.acc_meter.update(acc_pos+acc_neg, totals)

        self.tp_meter.update(tps)
        self.fp_meter.update(fps)
        self.tn_meter.update(tns)
        self.fn_meter.update(fns)
