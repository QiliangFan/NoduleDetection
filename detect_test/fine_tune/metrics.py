import torch
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