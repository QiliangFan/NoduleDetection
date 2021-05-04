from network.metrics import AccMeter, AverageMeter
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from typing import List
from functools import wraps

def reset(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        assert hasattr(self, "reset_meter"), f"expect `self` to have function reset_meter, but failed to get..."
        self.reset_meter()
        return func(self, *args, **kwargs)
    return wrapper

class AlexNet(LightningModule):
    
    def __init__(self, save_path: str = None):
        super().__init__()

        self.save_path = save_path
        self.init_meter()

        # network architecture
        self.conv1 = nn.Conv3d(1, 96, kernel_size=11, stride=4, )

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    @reset
    def training_epoch_end(self, outputs: List) -> None:
        pass

    def test_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    @reset
    def test_epoch_end(self, outputs: List) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        pass

    @torch.no_grad()
    @reset
    def validation_epoch_end(self, outputs: List) -> None:
        pass

    def init_meter(self):
        self.acc_meter = AccMeter()
        self.tp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.fn_meter = AverageMeter()

    def reset_meter(self):
        self.acc_meter.reset()
        self.tp_meter.reset()
        self.tn_meter.reset()
        self.fp_meter.reset()
        self.fn_meter.reset()