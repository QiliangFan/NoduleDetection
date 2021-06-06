from typing import Tuple
import torch
from torch.functional import Tensor
from pytorch_lightning import LightningModule
from torch.nn import Module
from metrics.dice import DiceCoefficient

class NetWork(LightningModule):

    def __init__(self, model: Module , **kwargs):
        super().__init__()

        self.model = model

        from torch.nn import BCEWithLogitsLoss
        self.loss = BCEWithLogitsLoss(pos_weight=torch.as_tensor(2))
        self.dice = DiceCoefficient()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        ct, nodule = batch
        out = self(ct)
        loss = self.loss(out, nodule)
        with torch.no_grad():
            dice = self.dice(out.clone(), nodule.clone())
            self.log_dict({"dice": dice}, prog_bar=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        ct, nodule = batch
        out = self(ct)
        dice = self.dice(out.clone(), nodule.clone())
        self.log_dict({"dice": dice}, prog_bar=True)
        return batch_idx

    def configure_optimizers(self):
        from torch.optim import Adam
        return Adam(self.parameters(), lr=1e-3)


