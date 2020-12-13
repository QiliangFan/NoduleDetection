from __future__ import print_function

import os
from multiprocessing import Pool
from metric.roc import get_roc_curve, plot_froc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from dataset import Luna16Dataset
from detection.model import DetectionModel, CenterLoss, SizeLoss
from datetime import datetime
import numpy as np


def show_progress(loss):
    print(f"{datetime.now(): %Y/%m/%d %H:%M:%S} total loss: {loss}")


def train(det_model, train_data, val_data, evaluate=False, total_epochs=5):
    cuda_count = torch.cuda.device_count()
    if cuda_count == 0:
        device = -1
    else:
        device = os.getpid() % cuda_count
    if device >= 0:
        torch.cuda.set_device(device)
    if torch.cuda.is_available():
        det_model.cuda()

    seg_optim = None
    det_optim = optim.SGD(det_model.parameters(),
                          lr=0.2, weight_decay=1e-4, momentum=0.6)
    seg_lr_scheduler = None
    det_lr_scheduler = optim.lr_scheduler.StepLR(det_optim, 1, gamma=0.75)

    center_loss = CenterLoss()
    sz_loss = SizeLoss()

    if not evaluate:
        for ep in range(total_epochs):
            for i, (ct, seg, nodule, sz, weight) in enumerate(train_data):
                ct = ct.reshape(1, 1, *ct.shape)

                if torch.cuda.is_available():
                    ct, seg, nodule, sz, weight = ct.cuda(), seg.cuda(), nodule.cuda(), sz.cuda(), weight.cuda()
                nodule_output, sz_output = det_model(ct)
                center_ls = center_loss(weight, nodule_output.squeeze(), nodule)
                size_ls = sz_loss(sz_output.squeeze(), sz)
                det_model.zero_grad()
                center_ls.backward()
                size_ls.backward()
                torch.nn.utils.clip_grad_norm_(det_model.parameters(), max_norm=10, norm_type=2.)
                det_optim.step()
                show_progress(center_ls.item() + size_ls.item())
                del ct, seg, nodule, sz, weight, center_ls, size_ls
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            det_lr_scheduler.step()

    fprs, tprs, thresholds = [], [], []
    with torch.no_grad():
        for i, (ct, seg, nodule, sz, weight) in enumerate(val_data):
            ct = ct.reshape(1, 1, *ct.shape)
            if torch.cuda.is_available():
                ct, seg, nodule, sz, weight = ct.cuda(), seg.cuda(), nodule.cuda(), sz.cuda(), weight.cuda()
            nodule_output, sz_output = det_model(ct)
            score = nodule_output.squeeze().clone()

            fpr, tpr, threshold = get_roc_curve(nodule.cpu().numpy().astype(np.float32),
                                                score.cpu().numpy().astype(np.float32))
            print(f"{datetime.now(): %H:%M:%S}")
            fprs.extend(fpr)
            tprs.extend(tpr)
            thresholds.extend(threshold)
            del ct, seg, nodule, sz, weight, nodule_output, sz_output, score
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return fprs, tprs, thresholds


class MultiTask:

    def __init__(self, ct_root: str, seg_root: str, nodule_root: str, sz_root: str, weight: str):
        # data
        data_root_list = [os.path.join(ct_root, dir) for dir in os.listdir(ct_root)]
        self.datasets = [Luna16Dataset(data_root, seg_root, nodule_root, sz_root, weight)
                         for data_root in data_root_list]

        self.train = []
        self.val = []
        self.num_image = 0
        for i in range(10):
            self.num_image += len(self.datasets[i])
            train_idx = list(range(0, 10))
            train_idx.pop(i)
            val_idx = i
            data = []
            for idx in train_idx:
                data.append(self.datasets[idx])
            data = ConcatDataset(data)
            self.train.append(data)
            self.val.append(self.datasets[val_idx])

    def cross_validation(self, test=False, evaluate=False, workers=None):
        if not test:  # multiprocessing
            task = []
            for i in range(10):
                seg_model = None
                det_model = DetectionModel()
                task.append([det_model, self.train[i], self.val[i], evaluate])
            with Pool(processes=workers) as pool:
                result = pool.starmap(train, task)
                fprs, tprs, thresholds = [], [], []
                for fpr, tpr, threshold in result:
                    fprs.extend(fpr)
                    tprs.extend(tpr)
                    thresholds.extend(threshold)
                fps, tprs = np.asarray(fprs), np.asarray(tprs)
                fps = fps / self.num_image
                save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saving")
                np.save(f"{os.path.join(save_path, 'fps')}", fps)
                np.save(f"{os.path.join(save_path, 'tpr')}", tprs)
                # np.save(f"{os.path.join(save_path, 'thresholds')}", thresholds)
            plot_froc_curve(fps, tprs)

        else:  # test
            for i in range(10):
                seg_model = None
                det_model = DetectionModel()
                train(det_model, self.train[i], self.val[i], evaluate)
