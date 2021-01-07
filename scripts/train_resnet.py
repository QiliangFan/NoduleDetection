"""
This script is used for FP reduction

balance: 50 epochs
imbalance: 100 epochs
"""
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"  # synchroniz
import json
from network.meter import AverageMeter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import numpy as np
from network.resnet import ResNet3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter
from torch.cuda.amp import autocast, GradScaler


# parameter
batch_size = 32  # was 32
# UNI_SIGN = "normal"
# UNI_SIGN = "deeper"  # represent for each network
# UNI_SIGN = "deeper_plus_plus"
UNI_SIGN = "deeper_ppp"
# UNI_SIGN = "original_1"
# UNI_SIGN = "imbalance"
# UNI_SIGN = "bottleneck"
# UNI_SIGN = "None"

def save_output(arr, pred, label, save_dir, k):
    for i, data in enumerate(arr):
        plt.figure()
        plt.imshow(data, cmap="bone")
        plt.savefig(os.path.join(save_dir, f"{k}_label{label}_pred{pred}_{i}.png"), bbox_inches="tight")
        plt.axis("off")
        plt.close()



class Data(Dataset):
    def __init__(self, patch_root: str, candidate_csv: str):
        super(Data, self).__init__()
        self.patch_file = glob(os.path.join(patch_root, "*.npy"))
        self.candidate_csv = pd.read_csv(candidate_csv)
        self.candidate_value = self.candidate_csv.values
 
    def __getitem__(self, idx):
        patch_file = self.patch_file[idx]
        patch_data = np.load(patch_file)
        patch_data = torch.as_tensor(patch_data, dtype=torch.float32)

        idx = int(os.path.splitext(os.path.basename(patch_file))[0])
        value = self.candidate_value[idx]
        sid, x, y, z, cls = value
        label = torch.as_tensor(int(cls))  # two-class classifier
        return patch_data, label

    def __len__(self):
        return len(self.patch_file)


def main(train=True, epochs=10):

    loss_meter = AverageMeter()
    tp_meter = AverageMeter()
    fp_meter = AverageMeter()
    tn_meter = AverageMeter()
    fn_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    acc_meter = AverageMeter()

    # train
    if train:
        model.train()
        for ep in range(epochs):
            _tqdm = tqdm(enumerate(train_data), total=len(train_data))
            for i, (patch, label) in _tqdm:
                idx = torch.where(label > 0)[0]
                if idx.numel() == 0:
                    continue
                additional_idx = (idx + 1)%len(patch)
                idx = torch.cat([idx, additional_idx])
                patch, label = patch[idx], label[idx]

                if torch.cuda.is_available():
                    patch, label = patch.cuda(), label.cuda()
                patch.unsqueeze_(dim=1)
                # print(patch.shape, label.shape, label)
                with autocast(enabled=False):
                    output = model(patch, True)
                    loss = criterion(output, label)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()

                # metric
                with torch.no_grad():
                    for out, target in zip(output.cpu(), label.cpu()):
                        cls = torch.argmax(out)
                        if cls == target:
                            acc_meter.update(1)
                            if cls == 1:
                                tp_meter.update(1)
                            else:
                                tn_meter.update(1)
                        else:
                            acc_meter.update(0)
                            if cls == 1:
                                fp_meter.update(1)
                            else:
                                fn_meter.update(1)
                    precision = tp_meter.sum / (tp_meter.sum + fp_meter.sum + 1e-6)
                    recall = tp_meter.sum / (tp_meter.sum + fn_meter.sum + 1e-6)
                    precision_meter.update(precision)
                    recall_meter.update(recall)
                    loss_val = loss.cpu().detach().item()
                    loss_meter.update(loss_val)
                    _tqdm.set_postfix(loss=loss_val, precision=precision, recall=recall, acc=acc_meter.avg)
                    if writer:
                        writer.add_scalar("loss", loss, global_step=i)

                del output, loss, patch, label
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # meter
            if writer:
                writer.add_scalars("train_metrics", tag_scalar_dict={
                            "precision": precision_meter.val,
                            "recall": recall_meter.val,
                            "acc": acc_meter.avg
                        }, global_step=ep)                    
            precision_meter.add_record(precision_meter.val)
            recall_meter.add_record(recall_meter.val)
            acc_meter.add_record()
            loss_meter.add_record()
            precision_meter.reset()
            recall_meter.reset()
            loss_meter.reset()
            tp_meter.reset()
            fp_meter.reset()
            tn_meter.reset()
            fn_meter.reset()
            acc_meter.reset()

            lr_scheduler.step()
            torch.save(model.state_dict(), model_path)
        loss_meter.plot(f"{UNI_SIGN}_resnet_trainloss.png")
        recall_meter.plot(f"{UNI_SIGN}_resnet_recall.png")
        precision_meter.plot(f"{UNI_SIGN}_resnet_precision.png")
        acc_meter.plot(f"{UNI_SIGN}_resnet_acc.png")
        loss_meter.save(f"{UNI_SIGN}_resnet_trainloss.txt")
        recall_meter.save(f"{UNI_SIGN}_resnet_recall.txt")
        precision_meter.save(f"{UNI_SIGN}_resnet_precision.txt")
        acc_meter.save(f"{UNI_SIGN}_resnet_acc.txt")

    loss_meter.reset(), loss_meter.clear_record()
    tp_meter.reset(), tp_meter.clear_record()
    fp_meter.reset(), tp_meter.clear_record()
    tn_meter.reset(), tn_meter.clear_record()
    fn_meter.reset(), fn_meter.clear_record()
    acc_meter.reset(), acc_meter.clear_record()
    precision_meter.reset(), precision_meter.clear_record()
    recall_meter.reset(), recall_meter.reset()

    # eval
    model.eval()
    with torch.no_grad():
        _tqdm = tqdm(test_data, total=len(test_data))
        t = 0

        for patch, label in _tqdm:
            if torch.cuda.is_available():
                patch, label = patch.cuda(), label.cuda()
            patch.unsqueeze_(dim=1)
            output = model(patch)

            # metric
            for i, (out, target) in enumerate(zip(output.cpu(), label.cpu())):
                if target == 0 and torch.argmax(out.cpu()) == 1:
                    save_output(patch[i].cpu().squeeze(), torch.argmax(out.cpu()), target.cpu(), os.path.join(project_path, "img"), t)
                    t += 1
                cls = torch.argmax(out)
                if cls == target:
                    acc_meter.update(1)
                    if cls == 1:
                        tp_meter.update(1)
                    else:
                        tn_meter.update(1)
                else:
                    acc_meter.update(0)
                    if cls == 1:
                        fp_meter.update(1)
                    else:
                        fn_meter.update(1)
                precision = tp_meter.sum / (tp_meter.sum + fp_meter.sum + 1e-6)
                recall = tp_meter.sum / (tp_meter.sum + fn_meter.sum + 1e-6)
                precision_meter.update(precision)
                recall_meter.update(recall)
            _tqdm.set_postfix(precision=precision_meter.val, recall=recall_meter.val, acc=acc_meter.avg)
            del output, patch, label
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        metrics = {
            "tp": tp_meter.sum,
            "fp": fp_meter.sum,
            "tn": tn_meter.sum,
            "fn": fn_meter.sum,
            "precision": precision_meter.val,
            "recall": recall_meter.val,
            "acc": acc_meter.avg
        }
        json.dump(metrics, open(f"{UNI_SIGN}_resnet_metric.json", "w"))


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = json.load(open(os.path.join(project_path, "config.json"), "rb"))
    candidate_patch_root = config["candidate_patch"]
    candidate_file = config["candidate_file"]
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{UNI_SIGN}_resnet.pk")
    result_dir = os.path.join(project_path, "result")
    log_dir = os.path.join(project_path, "log", f"resnet_{UNI_SIGN}_{datetime.now():%Y%m%d_%H%M%S}")

    patch_data = Data(patch_root=candidate_patch_root, candidate_csv=candidate_file)
    data_len = len(patch_data)
    train_data, test_data = torch.utils.data.random_split(patch_data, [data_len-data_len//10, data_len//10])
    # train_data, test_data = torch.utils.data.random_split(patch_data, [data_len//10, data_len-data_len//10])

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_data = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    
    model = ResNet3D()
    scaler = GradScaler()

    # writer = SummaryWriter(log_dir=log_dir)
    writer = None
    rand_input = torch.rand((1, 1, 32, 32, 32))
    # writer.add_graph(model, input_to_model=rand_input)

    weight = torch.as_tensor([0.2, 0.8]).half()
    if torch.cuda.is_available():
        model.cuda()
        weight = weight.cuda()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    criterion = nn.CrossEntropyLoss(weight)
    # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.4, weight_decay=1e-4)    
    # optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.4, weight_decay=1e-8)    # ppp
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.2, weight_decay=1e-8)    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.25)

    main(train=False, epochs=75)

