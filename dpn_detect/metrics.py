import torch
from torch import nn
from torch.functional import Tensor
import numpy as np

class AverageMeter:
    def __init__(self):
        self.total = 0
        self.num = 0
        self.avg = 0

    def update(self, val, n=1):
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
        pred, target = pred.numpy(), target.numpy()
        total = 0
        acc = 0
        if np.any(pred > 0.5):
            pred[pred > 0.5] = 1
        if np.any(pred <= 0.5):
            pred[pred <= 0.5] = 0
        for _pred, _target in zip(pred, target):
            if _pred == _target:
                acc += 1
            total += 1
        super().update(acc, total)

    def reset(self):
        super().reset()
        