import sys
import os

from torch.nn.modules.loss import BCELoss
project_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(project_path))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from glob import glob
from datetime import datetime
from .dataset import Data
from .resnet import Resnet3D
from torch.optim import SGD, Adam, lr_scheduler
from network.meter import AverageMeter


def precision_recall(output: torch.Tensor, label: torch.Tensor):
    output, label = output.cpu(), label.cpu()
    # tp
    arr = output > 0.5
    arr = arr.long()
    tp = (arr * label).sum()
    
    # tn
    arr = output <= 0.5
    target = label == 0
    arr = arr.long()
    tn = (arr * target).sum()

    # fp
    arr = output > 0.5
    target = label == 0
    arr = arr.long()
    fp = (arr * target).sum()

    # fn
    arr = output <= 0.5
    target = label == 1
    arr = arr.long()
    fn = (arr * target).sum()

    # acc
    arr = output > 0.5
    arr = arr.long()
    acc_num = (arr == target).sum() 
    total = target.numel()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return precision, recall, acc_num, total


def train(train_data: Dataset, test_data: Dataset, epochs=75):
    recall_meter, precision_meter, acc_meter = AverageMeter(), AverageMeter(), AverageMeter()
    model_path = os.path.jon(project_path, "test", "model.pt")
    model = Resnet3D(1, 1, verbose=True)  # use BCE loss
    if os.path.exists(model_path):
       model.load_state_dict(torch.load(model_path))
    critertion = nn.BCELoss()
    # optim = SGD(model.parameters(), lr=0.001, momentum=0.2, weight_decay=1e-4)
    optim = Adam(model.parameters())
    LR_schduler = lr_scheduler.StepLR(optim, step_size=1, gamma=0.4)

    # train
    _tqdm = tqdm(train_data, total=len(train_data))
    for ep in range(epochs):
        for data, label in _tqdm:
            data.squeeze_(dim=1)
            output = model(data)
            loss = critertion(output, label)
            with torch.no_grad():
                precision, recall, acc_num, total = precision_recall(output, label)
                precision_meter.update(precision)
                recall_meter.update(recall)
                acc_meter.update(acc_num, total)
                writer.add_scalars("iters/train", tag_scalar_dict={
                    "precision": precision,
                    "recall": recall,
                    "acc": acc_meter.avg,
                    "loss": loss.item()
                })
            optim.zero_grad()
            loss.backward()
            optim.step()
        LR_schduler.step()
        writer.add_scalars("batch/train(avg)", tag_scalar_dict={
            "precision": precision_meter.avg,
            "recall": recall_meter.avg,
            "acc": acc_meter.acc
        })
        precision_meter.reset()
        recall_meter.reset()
        acc_meter.reset()
    

    if not os.path.exists(model_path):
        os.mkdir(model_path)
        torch.save(model.state_dict(), model_path)

    # eval
    _tqdm = tqdm(test_data, total=len(test_data))
    with torch.no_grad():
        for data, label in _tadm:
            data.squeeze_(dim=1)
            output = model(data)
            precision, recall, acc_num, total = precision_recall(output, label)
            writer.add_scalars("iters/test", tag_scalar_dict={
                "precision": precision,
                "recall": recall,
                "acc": acc_meter.avg,
            })

def main():
    # Dataset (4:1 - train test)
    pos_list = glob(os.path.join(pos_root, "*.npy"))
    neg_list = glob(os.path.join(neg_root, "*.npy"))
    pos_len, neg_len = len(pos_list), len(neg_list)
    aug_list = glob(os.path.join(augmentation_root, "*,npy"))

    pos_data = Data(pos_list)
    neg_data = Data(neg_list)
    train_pos_data, test_pos_data = random_split(pos_data, [pos_len-pos_len//5, pos_len//5])
    train_neg_data, test_neg_data = random_split(neg_data, [neg_len-neg_len//5, neg_len//5])
    aug_data = Data(aug_list)
    train_data = ConcatDataset([train_pos_data, train_neg_data, aug_data])
    test_data = ConcatDataset([test_pos_data, test_neg_data])

    train_data = DataLoader(train_data, shuffle=True, batch_size=32, num_workers=4, pin_memory=True)
    test_data = DataLoader(test_data, shuffle=True, batch_size=32, num_workers=4, pin_memory=True)
    train(train_data, test_data)



if __name__ == "__main__":
    
    augmentation_root = "/home/maling/fanqiliang/data/tmp/augmented_data"
    pos_root = "/home/maling/fanqiliang/data/tmp/patch/1"
    neg_root = "/home/maling/fanqiliang/data/tmp/patch/0"
    log_root = os.path.join(project_path, "log")
    
    run_time = f"{datetime.now():%Y%m%d_%H%M%S}"
    writer = SummaryWriter(log_dir=os.path.join(log_root, f"{run_time}", "resnet"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
