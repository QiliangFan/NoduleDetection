from __future__ import print_function

import os
from multiprocessing import Pool, Queue, Manager
from metric.roc import get_roc_curve, plot_froc_curve
from sklearn.metrics import precision_recall_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from dataset import Luna16Dataset
from detection.model import DetectionModel, CenterLoss, SizeLoss
from datetime import datetime
import numpy as np
import queue

THRESHOLD = 0.2

def show_progress(ct_loss, size_loss):
    loss = ct_loss + size_loss
    print(f"\r {datetime.now(): %Y/%m/%d %H:%M:%S} total loss: {loss}"
          f"\tcenter loss : {ct_loss} - size loss: {size_loss}", end="\r")


def train(det_model: nn.Module, train_data, val_data, q: queue.Queue, evaluate=False, det_model_path=None, total_epochs=5, order=0):
    cuda_count = torch.cuda.device_count()
    if cuda_count == 0:
        device = -1
    else:
        device = order % cuda_count
    if device >= 0:
        torch.cuda.set_device(device)
    if torch.cuda.is_available():
        det_model.cuda()
    if os.path.exists(det_model_path):
        det_model.load_state_dict(torch.load(det_model_path))

    seg_optim = None
    det_optim = optim.SGD(det_model.parameters(),
                          lr=0.2, weight_decay=1e-4, momentum=0.6)
    seg_lr_scheduler = None
    det_lr_scheduler = optim.lr_scheduler.StepLR(det_optim, 1, gamma=0.75)

    center_loss = CenterLoss()
    sz_loss = SizeLoss()

    if not evaluate:
        det_model.train()
        for ep in range(total_epochs):
            for i, (ct, seg, nodule, sz, weight) in enumerate(train_data):
                ct = ct.reshape(1, 1, *ct.shape)

                if torch.cuda.is_available():
                    ct, seg, nodule, sz, weight = ct.cuda(), seg.cuda(), nodule.cuda(), sz.cuda(), weight.cuda()
                nodule_output, sz_output = det_model(ct)
                center_ls = center_loss(weight, nodule_output.squeeze(), nodule)
                size_ls = sz_loss(sz_output.squeeze(), sz)
                det_model.zero_grad()
                loss = 0.1 * center_ls + 0.01 * size_ls
                loss.backward()
                # center_ls.backward()
                # size_ls.backward()
                torch.nn.utils.clip_grad_norm_(det_model.parameters(), max_norm=10, norm_type=2.)
                det_optim.step()
                show_progress(center_ls.item(), size_ls.item())
                del ct, seg, nodule, sz, weight, center_ls, size_ls, nodule_output, sz_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            det_lr_scheduler.step()
        if det_model_path:
            torch.save(det_model.state_dict(), det_model_path)
    else:
        det_model.eval()
        labels, scores = [], []
        with torch.no_grad():
            for i, (ct, seg, nodule, sz, weight) in enumerate(val_data):
                ct = ct.reshape(1, 1, *ct.shape)
                if torch.cuda.is_available():
                    ct, seg, nodule, sz, weight = ct.cuda(), seg.cuda(), nodule.cuda(), sz.cuda(), weight.cuda()
                nodule_output, sz_output = det_model(ct)
                score = nodule_output.squeeze().clone()

                print(f"\r {datetime.now(): %H:%M:%S}", end="\r")
                with torch.no_grad():
                    idx = torch.where(score >= THRESHOLD)
                    label = nodule.cpu()[idx].reshape(-1).tolist()
                    sc = score.cpu()[idx].reshape(-1).tolist()

                    q.put([label, sc])
                    # labels.extend(label)
                    # scores.extend(sc)
                
                # precision, recall, threshold = precision_recall_curve(label, sc)
                # f1_score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-6, a_max=None)
                # idx = np.nanargmax(f1_score)
                # precisions.append(precision[idx]), recalls.append(recall[idx])

                # if 4 <= i <= 10:
                #     ct.squeeze_()
                #     import matplotlib.pyplot as plt
                #     if not os.path.exists(f"/home/maling/fanqiliang/img/{i}"):
                #         os.mkdir(f"/home/maling/fanqiliang/img/{i}")
                #     for k, arr in enumerate(ct.cpu()):
                #         fig, ax = plt.subplots(2, 2)
                #         ax[0][0].imshow(ct[k].cpu(), cmap="bone")
                #         ax[0][0].set_title("raw ct data")
                #         ax[0][1].imshow(score[k].cpu(), cmap="bone")
                #         ax[0][1].set_title("score")
                #         ax[1][0].imshow(nodule[k].cpu())
                #         ax[1][0].set_title("nodule label")
                #         im = ax[1][1].imshow(ct[k].cpu(), cmap="jet")
                #         ax[1][1].set_title("heat map")
                #         plt.colorbar(im, pad=0.02)
                #         plt.savefig(f"/home/maling/fanqiliang/img/{i}/img_{k}.png")
                #         plt.close(fig)

                del ct, seg, nodule, sz, weight, nodule_output, sz_output, score
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
    print("\nprocess finished, prepare to return.")
    return 0
        # return labels, scores


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

    def cross_validation(self, test=False, evaluate=False):
        num_fold = 10
        total_eopch = 1

        if not test:  # multiprocessing
            task = []
            q_list = []
            for i in range(10):
                q_list.append(Manager().Queue(200))
            for i in range(num_fold):
                seg_model = None
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", f"model_det_{i}.pk")
                det_model = DetectionModel()
                det_model.cpu()
                task.append([det_model, self.train[i], self.val[i], q_list[i], evaluate, model_path, total_eopch, i])
            with Pool(processes=len(task)) as pool:
                result = pool.starmap(train, task)
                pool.close()
                pool.join()
                print("process pool closed!")

                if evaluate:
                    result = []
                    for i in range(num_fold):
                        while not q_list[i].empty():
                            result.append(q_list[i].get())
                    result = np.concatenate(result, axis=1)
                    labels, scores = result
                    fps, tpr, threshold = get_roc_curve(labels, scores)
                    plot_froc_curve(fps, tpr, self.num_image)

                
            # plot_froc_curve(fps, tprs)

        else:  # test
            for i in range(10):
                seg_model = None
                det_model = DetectionModel()
                train(det_model, self.train[i], self.val[i], evaluate)
