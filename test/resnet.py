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
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

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
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += res
        x = self.relu2(x)
        return x
                        

class Resnet3D(LightningModule):
    def __init__(self, in_channel, num_classes, verbose=False, dropout=0.5, save_root=None):
        super(Resnet3D, self).__init__()
        self.save_root = save_root
        self.verbose = verbose
        # 1*24*40*40

        num_feature = 2
        self.block1 = nn.Conv3d(in_channel,
                                num_feature, 
                                (7, 7, 7),
                                stride=1,
                                padding=3)  # 16*48*48*48
        self.block2 = nn.Sequential(
            ResBlock(num_feature, num_feature*2, stride=2),  # 16*24
            nn.Dropout3d(p=dropout),
            ResBlock(num_feature*2, num_feature*2), 
            nn.Dropout3d(p=dropout),
            # nn.MaxPool3d((2, 2, 2), stride=2),  # 16*12
        )
        num_feature *= 2
        self.block3 = nn.Sequential(
            ResBlock(num_feature, num_feature*2, stride=2), # 32*12
            nn.Dropout3d(p=dropout),
            ResBlock(num_feature*2, num_feature*2),
            nn.Dropout3d(p=dropout),
            # nn.MaxPool3d((2, 2, 2), stride=2),  # 32*6
        )
        num_feature *= 2
        self.block4 = nn.Sequential(
            ResBlock(num_feature, num_feature*2, stride=2), # 64*6
            nn.Dropout3d(p=dropout),
            ResBlock(num_feature*2, num_feature*2),
            nn.Dropout3d(p=dropout),
        )
        num_feature *= 2
        self.block5 = nn.Sequential(
            ResBlock(num_feature, num_feature*2, stride=2), # 128*3
            nn.Dropout(p=dropout),
            ResBlock(num_feature*2, num_feature*2),
            nn.Dropout(p=dropout),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        num_feature *= 2
        self.fc = Sequential(
            nn.Flatten(),
            nn.Linear(num_feature, num_feature),
            nn.Dropout(p=dropout),
            nn.Linear(num_feature, num_classes),
            # nn.Sigmoid()
        )

        # criterion and metric
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([5, 1]))
        # self.criterion = nn.BCELoss()
        self.acc_meter = AverageMeter()
        self.tp_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fn_meter = AverageMeter()

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)
        return x
        
    def training_step(self, batch: torch.Tensor, batch_idx):
        self.batch_idx = batch_idx
        files, data, target = batch
        out = self(data)
        loss = self.criterion(out, target)
        return loss


    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.acc_meter.reset()
        self.tp_meter.reset()
        self.fp_meter.reset()
        self.tn_meter.reset()
        self.fn_meter.reset()

    def validation_step(self, batch, batch_idx):
        self.test_operation(batch, batch_idx)
        return batch_idx

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        precision = self.tp_meter.sum / (self.tp_meter.sum + self.fp_meter.sum + 1e-6)
        recall = self.tp_meter.sum / (self.tp_meter.sum + self.fn_meter.sum + 1e-6)
        self.log_dict({
            "precision": precision,
            "recall": recall,
            "accuracy": self.acc_meter.avg,
        }, prog_bar=True)
        
        self.log_dict({
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

    def test_step(self, batch, batch_idx):
        self.test_operation(batch, batch_idx, save=True)
        return batch_idx

    def test_epoch_end(self, outputs: List[Any]) -> None:
        precision = self.tp_meter.sum / (self.tp_meter.sum + self.fp_meter.sum + 1e-9)
        recall = self.tp_meter.sum / (self.tp_meter.sum + self.fn_meter.sum + 1e-9)
        self.log_dict({
            "precision": precision,
            "recall": recall,
            "accuracy": self.acc_meter.avg,
        }, prog_bar=True)

        self.log_dict({
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
                        weight_decay=1e-4,
                        eps=1e-3,
                        amsgrad=False
                        # momentum=0.2, 
                        # weight_decay=1e-4
                        )
        # opt = optim.RMSprop(self.parameters(), lr=1e-4, weight_decay=1e-4, eps=1e-3)
        stepLr = optim.lr_scheduler.StepLR(opt, 1, gamma=0.9)
        # return opt
        return {
            "optimizer": opt,
            "lr_scheduler": stepLr
        }

    @torch.no_grad()
    def precision_recall(self, output: torch.Tensor, target: torch.Tensor):
        # if hasattr(output, "cpu") and hasattr(target, "cpu"):
        #     output, target = output.cpu(), target.cpu()
        cp_out, cp_target = output.detach(), target.detach()
        tps = 0
        fps = 0
        tns = 0
        fns = 0
        total_acc = 0
        total = 0

        for _out, _tg in zip(cp_out, cp_target):
            if _out > 0.5:
                pred = 1
            else:
                pred = 0
            if _tg > 0:
                label = 1
            else:
                label = 0
            
            total += 1
            if pred == label:
                total_acc += 1
                if label == 1:
                    tps += 1
                else:
                    tns += 1
            else:
                if label == 1:
                    fps += 1
                else:
                    fns += 1

        self.acc_meter.update(total_acc, total)

        self.tp_meter.update(tps)
        self.fp_meter.update(fps)
        self.tn_meter.update(tns)
        self.fn_meter.update(fns)

    @torch.no_grad()
    def test_operation(self, batch, batch_idx, save=False):
        self.batch_idx = batch_idx
        files, data, target = batch
        out: Tensor = self(data)
        self.precision_recall(out, target)
        if save:
            with open(os.path.join(self.save_root, "output.csv") if self.save_root else "output.csv", "a") as fp:
                for v, l in zip(out[:, 1].cpu().squeeze().tolist(), target[:, 1].cpu().squeeze().tolist()):
                    print(v, l, sep=",", file=fp)


