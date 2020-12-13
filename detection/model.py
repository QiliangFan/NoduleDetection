"""
A detection model for test
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def make_conv3d(inplane:int, plane:int, activation=True):
    if activation:
        return nn.Sequential(
            nn.Conv3d(inplane, plane, (3, 3, 3), stride=1, padding=1),
            nn.InstanceNorm3d(plane),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            nn.Conv3d(inplane, plane, (3, 3, 3), stride=1, padding=1),
        )


class DetectionModel(nn.Module):
    """
    center identification
    size regression
    """
    def __init__(self):
        super(DetectionModel, self).__init__()

        self.center_identify = nn.Sequential(
            make_conv3d(1, 4),
            make_conv3d(4, 4),
            make_conv3d(4, 4),
            make_conv3d(4, 4),
            nn.Conv3d(4, 1, (1, 1, 1)),
            nn.Sigmoid()
        )

        self.size_regression = nn.Sequential(
            make_conv3d(1, 4),
            make_conv3d(4, 4),
            make_conv3d(4, 4),
            make_conv3d(4, 4),
            nn.Conv3d(4, 3, (1, 1, 1)),
        )

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return self.center_identify(x), self.size_regression(x)


class CenterLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(CenterLoss, self).__init__()
        self.alpha = 2
        self.beta = 4

    def forward(self, weight: Tensor, score: Tensor, nodule: Tensor) -> Tensor:
        non_center_idx = torch.where(nodule == 0)
        center_idx = torch.where(nodule == 1)

        score_center = torch.as_tensor(score[center_idx] if center_idx[0].numel() else 0)

        nodule_loss = torch.mean((1 - score_center)**self.alpha * torch.log(score_center + 1e-6))
        non_nodule_loss = torch.mean((1-weight[non_center_idx])**self.beta *
                                     score[non_center_idx]**self.beta * torch.log(1 - score[non_center_idx]))
        nodule_loss, non_nodule_loss = nodule_loss.to(score.device), non_nodule_loss.to(score.device)
        loss = -1 * torch.mean(nodule_loss + non_nodule_loss)
        loss = loss.to(score.device)
        return loss


class SizeLoss(nn.Module):
    def __init__(self):
        super(SizeLoss, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, sz_output: Tensor, sz: Tensor) -> Tensor:
        sz_output, sz = sz_output.reshape(sz_output.shape[0], -1), sz.reshape(sz.shape[0], -1)
        loss = self.loss(sz_output, sz)
        loss = loss.to(sz_output.device)
        return loss
