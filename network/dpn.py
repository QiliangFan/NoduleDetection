""" PyTorch implementation of DualPathNetworks
Based on original MXNet implementation https://github.com/cypw/DPNs with
many ideas from another PyTorch implementation https://github.com/oyam/pytorch-DPNs.

This implementation is compatible with the pretrained weights
from cypw's MXNet implementation.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.metrics import AccMeter, AverageMeter
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from collections import OrderedDict
from pytorch_lightning import LightningModule
from network.adaptive_avgmax_pool import adaptive_avgmax_pool3d


__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107', "getdpn"]


model_urls = {
    'dpn68':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68-66bebafa7.pth',
    'dpn68b-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn68b_extra-84854c156.pth',
    'dpn92-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn92_extra-b040e4a9b.pth',
    'dpn98':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn98-5b90dec4d.pth',
    'dpn131':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn131-71dfe43e0.pth',
    'dpn107-extra':
        'https://github.com/rwightman/pytorch-dpn-pretrained/releases/download/v0.1/dpn107_extra-1ac7121e2.pth'
}


def getdpn(**kwargs):
    model = DPN(
        small=True, num_init_features=2, k_r=2, groups=1,
        k_sec=(2, 2, 3, 2), inc_sec=(2, 4, 8, 16),
        test_time_pool=False, **kwargs)
    return model


def dpn68(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-68 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn68']))
    return model


def dpn68b(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-68b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        small=True, num_init_features=10, k_r=128, groups=32,
        b=True, k_sec=(3, 4, 12, 3), inc_sec=(16, 32, 32, 64),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_state_dict_from_url(model_urls['dpn68b-extra']))
    return model


def dpn92(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-92 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=64, k_r=96, groups=32,
        k_sec=(3, 4, 20, 3), inc_sec=(16, 32, 24, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_state_dict_from_url(model_urls['dpn92-extra']))
    return model


def dpn98(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-98 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=96, k_r=160, groups=40,
        k_sec=(3, 6, 20, 3), inc_sec=(16, 32, 32, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn98']))
    return model


def dpn131(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-131 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=128, k_r=160, groups=40,
        k_sec=(4, 8, 28, 3), inc_sec=(16, 32, 32, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['dpn131']))
    return model


def dpn107(pretrained=False, test_time_pool=False, **kwargs):
    """Constructs a DPN-107 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet-1K
        test_time_pool (bool): If True, pools features for input resolution beyond
            standard 224x224 input with avg+max at inference/validation time

        **kwargs : Keyword args passed to model __init__
            num_classes (int): Number of classes for classifier linear layer, default=1000
    """
    model = DPN(
        num_init_features=128, k_r=200, groups=50,
        k_sec=(4, 8, 20, 3), inc_sec=(20, 64, 64, 128),
        test_time_pool=test_time_pool, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_state_dict_from_url(model_urls['dpn107-extra']))
    return model


class CatBnAct(nn.Module):
    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm3d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv3d(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride,
                 padding=0, groups=1, activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv3d, self).__init__()
        self.bn = nn.BatchNorm3d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv3d(in_chs, out_chs, kernel_size,
                              stride, padding, groups=groups, bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):
    def __init__(self, num_init_features, kernel_size=7,
                 padding=3, activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv3d(
            1, num_init_features, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):
    def __init__(
            self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, groups, block_type='normal', b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        self.dropout = nn.Dropout3d(inplace=False)
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            # Using different member names here to allow easier parameter key matching for conversion
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv3d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=2)
            else:
                self.c1x1_w_s1 = BnActConv3d(
                    in_chs=in_chs, out_chs=num_1x1_c + 2 * inc, kernel_size=1, stride=1)
        self.c1x1_a = BnActConv3d(
            in_chs=in_chs, out_chs=num_1x1_a, kernel_size=1, stride=1)
        self.c3x3_b = BnActConv3d(
            in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3,
            stride=self.key_stride, padding=1, groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv3d(
                num_3x3_b, num_1x1_c, kernel_size=1, bias=False)
            self.c1x1_c2 = nn.Conv3d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv3d(
                in_chs=num_3x3_b, out_chs=num_1x1_c + inc, kernel_size=1, stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.dropout(x_in)
        x_in = self.c1x1_a(x_in)
        x_in = self.dropout(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(LightningModule):
    def __init__(self,
                 small=False,
                 num_init_features=64,
                 k_r=96,
                 groups=32,
                 b=False,
                 k_sec=(3, 4, 20, 3),
                 inc_sec=(16, 32, 24, 128),
                 num_classes=1,
                 test_time_pool=False,
                 save_dir: str = None):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = OrderedDict()

        if save_dir is not None:
            self.save_dir = save_dir
        else:
            self.save_dir = None

        # conv1
        if small:
            blocks['conv1_1'] = InputBlock(
                num_init_features, kernel_size=3, padding=1)
        else:
            blocks['conv1_1'] = InputBlock(
                num_init_features, kernel_size=7, padding=3)

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv2_1'] = DualPathBlock(
            num_init_features, r, r, bw, inc, groups, 'proj', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks['conv2_' + str(i)] = DualPathBlock(in_chs,
                                                      r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv3_1'] = DualPathBlock(
            in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks['conv3_' + str(i)] = DualPathBlock(in_chs,
                                                      r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv4_1'] = DualPathBlock(
            in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks['conv4_' + str(i)] = DualPathBlock(in_chs,
                                                      r, r, bw, inc, groups, 'normal', b)
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks['conv5_1'] = DualPathBlock(
            in_chs, r, r, bw, inc, groups, 'down', b)
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks['conv5_' + str(i)] = DualPathBlock(in_chs,
                                                      r, r, bw, inc, groups, 'normal', b)
            in_chs += inc
        blocks['conv5_bn_ac'] = CatBnAct(in_chs)

        self.features = nn.Sequential(blocks)

        # Using 1x1 conv for the FC layer to allow the extra pooling scheme
        self.classifier = nn.Conv3d(
            in_chs, num_classes, kernel_size=1, bias=True)

        # criterion
        from torch.nn import BCELoss
        self.bce_loss = BCELoss()
        self.acc = AccMeter()

        self.tp_meter = AverageMeter()
        self.fp_meter = AverageMeter()
        self.tn_meter = AverageMeter()
        self.fn_meter = AverageMeter()

    def forward(self, x):
        x = self.features(x)
        if not self.training and self.test_time_pool:
            x = F.avg_pool3d(x, kernel_size=7, stride=1)
            out = self.classifier(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool3d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool3d(x, pool_type='avg')
            out = self.classifier(x)
        out = torch.sigmoid(out)
        return out.view(out.size(0), -1)

    def training_step(self, batch, batch_idx):
        ct, nodule = batch
        out = self(ct)
        loss = self.bce_loss(out, nodule)
        with torch.no_grad():
            self.acc.update(out.detach().cpu(), nodule.detach().cpu())
        self.log("acc", self.acc.avg, prog_bar=True)
        self.log("output_max", torch.max(out).item(), prog_bar=True)
        self.log("output_min", torch.min(out).item(), prog_bar=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.acc.reset()

    @torch.no_grad()
    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        ct, nodule = batch
        out = self(ct)

        if self.save_dir is not None:
            import os
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            with open(os.path.join(self.save_dir, "output.txt"), "a") as fp:
                for _out, _nodule in zip(out.cpu(), nodule.cpu()):
                    if _out.item() > 0.5 and _nodule == 1:  # TP
                        self.tp_meter.update(1, 1)
                        self.fp_meter.update(0, 1)
                        self.tn_meter.update(0, 1)
                        self.fn_meter.update(0, 1)
                    elif _out.item() > 0.5 and _nodule == 0:  # FP
                        self.tp_meter.update(0, 1)
                        self.fp_meter.update(1, 1)
                        self.tn_meter.update(0, 1)
                        self.fn_meter.update(0, 1)
                    elif _out.item() <= 0.5 and _nodule == 1:  # FN
                        self.tp_meter.update(0, 1)
                        self.fp_meter.update(0, 1)
                        self.tn_meter.update(0, 1)
                        self.fn_meter.update(1, 1)
                    else:  # TN
                        self.tp_meter.update(0, 1)
                        self.fp_meter.update(0, 1)
                        self.tn_meter.update(1, 1)
                        self.fn_meter.update(0, 1)
                    print(_out.item(), _nodule.item(), sep=",", file=fp)
            self.log_dict({
                "tp": self.tp_meter.total,
                "fp": self.fp_meter.total,
                "tn": self.tn_meter.total,
                "fn": self.fn_meter.total,
            }, prog_bar=True)
        self.acc.update(out.detach().cpu(), nodule.detach().cpu())
        self.log("acc", self.acc.avg, prog_bar=True)
        return batch_idx

    def test_epoch_end(self, outputs):
        tp = self.tp_meter.total
        tn = self.tn_meter.total
        fp = self.fp_meter.total
        fn = self.fn_meter.total

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        self.log_dict({"precision": precision, "recall": recall}, prog_bar=True)

        with open("metrics.txt", "a") as fp:
            import json
            result = self.trainer.logged_metrics
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].item()
            result = json.dumps(result, indent=4)
            print(result, file=fp)
        
        self.acc.reset()
        self.tp_meter.reset()
        self.tn_meter.reset()
        self.fp_meter.reset()
        self.fn_meter.reset()

    def configure_optimizers(self):
        from torch.optim import SGD

        sgd = SGD(self.parameters(), lr=1e-3, momentum=0.1, weight_decay=1e-4)
        return sgd
