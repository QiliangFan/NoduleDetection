import torch
from torch import nn
from torch.functional import Tensor


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: Tensor, targets: Tensor):
        # 计算每张图单独的loss
        num = torch.sum(torch.mul(pred, targets), dim=1) + 1
        den = torch.sum(pred.pow(2) + targets.pow(2), dim=1) + 1
        loss = 1 - num / den
        return loss.mean()
