import math
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        pass

    def forward(self, output, target):
        """
        :param output: (N, C, ...)
        :param target: (N, C, ...)
        """
        output = output.reshape(output.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        intersection = (output* (target*1000)).sum(dim=1)
        union = torch.sum(output, dim=1) + torch.sum(target, dim=1)
        return 1 - torch.mean((2*intersection +1) / (union+1))