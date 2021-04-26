import torch
import numpy as np
from torch import nn
from torch.functional import Tensor


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: Tensor, targets: Tensor):
        # 计算每张图单独的loss
        num = 2 * torch.sum(torch.mul(pred, targets), dim=1) + 1
        den = torch.sum(pred + targets, dim=1) + 1
        loss = 1 - num / den
        return loss.mean()

class DiceCoefficient(nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, pred: Tensor, targets: Tensor):
        num = 2 * torch.sum(torch.mul(pred, targets), dim=1) + 1
        den = torch.sum(pred + targets, dim=1) + 1
        coe = num / den
        return coe.mean()

class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred: Tensor, target: Tensor):
        total = 0
        acc = 0
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        for _pred, _target in zip(pred, target):
            if _pred == _target:
                acc += 1
            total += 1

        return 

class AverageMeter:
    def __init__(self):
        self.total = 0
        self.num = 0
        self.avg = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = int(val.item())
        elif isinstance(val, np.ndarray):
            val = int(val.item())
        else:
            val = int(val)
        self.total += val
        self.num += n
        self.avg = self.total / self.num

    def reset(self):
        self.total = 0
        self.num = 0
        self.avg = 0

class AccMeter(AverageMeter):
    def __init__(self):
        super().__init__()

    def update(self, pred: Tensor, target: Tensor):
        total = 0
        acc = 0
        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0
        for _pred, _target in zip(pred, target):
            if _pred == _target:
                acc += 1
            total += 1
        super().update(acc, total)

    def reset(self):
        super().reset()
        