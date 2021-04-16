import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        num = torch.sum(torch.mul(pred, target), dim=1) + 1
        den = torch.sum(pred.pow(2) + target.pow(2), dim=1) + 1
        loss = 1 - num / den
        return loss.mean()


class DiceCoefficient(nn.Module):

    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        num = torch.sum(torch.mul(pred, target), dim=1) + 1
        den = torch.sum(pred + target, dim=1) + 1
        result = 2 * num / den
        return result.mean()
