import torch
from torch import nn
from torch import Tensor

class DiceCoefficient(nn.Module):
    def __init__(self):
        super(DiceCoefficient, self).__init__()

    def forward(self, pred: Tensor, targets: Tensor):
        pred, targets = pred.detach(), targets.detach()
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        num = 2 * torch.sum(torch.mul(pred, targets), dim=1)
        den = torch.sum(pred + targets, dim=1) + 1e-6
        coe = num / den
        return coe.mean()