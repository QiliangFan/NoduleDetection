import torch
from torch import nn
from torch.functional import Tensor


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred: Tensor, targets: Tensor):
        # 计算每张图单独的loss
        dice_loss = 1 - (2 * torch.sum(pred * targets, dim=1) + 1) / ((torch.sum(pred, dim=1) + torch.sum(targets, dim=1) + 1))
        dice_loss = torch.mean(dice_loss)
        return dice_loss
