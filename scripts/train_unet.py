import sys

from torch._C import dtype
sys.path.append("/home/maling/fanqiliang/projects/NoduleDetection")

from typing import List, Tuple, Union
from network.unet import Unet3D
from network.loss import DiceLoss
from network.meter import AverageMeter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from glob import glob
from tqdm import tqdm
import json
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard.writer import SummaryWriter


# parameter
os.environ["CUDA_VISIBLE_DEVICES"]="0"
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = json.load(open(os.path.join(project_path, "config.json"), "rb"))
slice_thickness = config["slice_thickness"]
UNI_SIGN = "normal"
# UNI_SIGN = "deeper"
# UNI_SIGN = "deeper_slice"

class Data(Dataset):
    def __init__(self, ct_dirs: Union[List, Tuple], nodule_dirs: Union[List, Tuple]):
        super(Data, self).__init__()
        assert ct_dirs and nodule_dirs
        self.ct = []
        self.nodule = []
        for ct_dir, nodule_dir in tqdm(zip(ct_dirs, nodule_dirs)):
            ct_files = glob(os.path.join(ct_dir, "*.mhd"))
            nodule_files = [os.path.join(nodule_dir, os.path.basename(_)) for _ in ct_files]
            assert len(nodule_files) and len(ct_files), "There should be some images!"
            self.ct.extend(ct_files)
            self.nodule.extend(nodule_files)
            assert len(self.ct) == len(self.nodule), "size should be equal!"

    def __getitem__(self, idx):
        ct = sitk.GetArrayFromImage(sitk.ReadImage(self.ct[idx]))
        nodule = sitk.GetArrayFromImage(sitk.ReadImage(self.nodule[idx]))
        return torch.as_tensor(ct, dtype=torch.float32), \
               torch.as_tensor(nodule, dtype=torch.long)

    def __len__(self):
        return len(self.ct)

def gen_slices(ct: torch.Tensor, nodule: torch.Tensor):
    ct.squeeze_(), nodule.squeeze_()
    assert len(ct) == len(nodule)
    ct_list = []
    nodule_list = []
    for i in range(0, len(ct), slice_thickness):
        if i + slice_thickness - 1 < len(ct):
            ct_list.append(ct[i:i+slice_thickness])
            nodule_list.append(nodule[i:i+slice_thickness])
        else:
            zero_pd = torch.zeros((slice_thickness - (len(ct)-i), ct.shape[1], ct.shape[2]))
            ct_list.append(torch.cat([ct[i:i+slice_thickness],
                                      zero_pd], dim=0))
            nodule_list.append(torch.cat([nodule[i:i+slice_thickness], 
                                      zero_pd], dim=0))
    return ct_list, nodule_list


def plot(ct, nodule, output, save_path, idx):
    output = torch.sigmoid(output)
    for i, (slice_ct, slice_nodule, slice_output) in enumerate(zip(ct, nodule, output)):
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].set_title("original data")
        ax[0].imshow(slice_ct, cmap="bone")

        ax[2].set_title("nodule target")
        ax[2].imshow(slice_nodule, cmap="bone")

        ax[1].set_title("score")
        im = ax[1].imshow(slice_output, cmap="jet")

        fig.colorbar(im, ax=ax[1], pad=0.03, shrink=1)
        fig.tight_layout(pad=1.08, h_pad=0.5)
        plt.savefig(os.path.join(save_path, f"{idx}_{i}.png"), bbox_inches="tight")
        plt.close(fig)


def main(ct_root, nodule_root, epochs=1, train=True):
    ct_dirs = glob(os.path.join(ct_root, "subset*"))
    nodule_dirs = [os.path.join(nodule_root, os.path.basename(_)) for _ in ct_dirs]
    train_data = Data(ct_dirs[:9], nodule_dirs[:9])
    test_data = Data(ct_dirs[9:10], nodule_dirs[9:10])

    train_data = DataLoader(train_data, batch_size=1, num_workers=1, shuffle=True)
    test_data = DataLoader(test_data, batch_size=1, num_workers=1, shuffle=True)

    train_meter = AverageMeter()
    test_meter = AverageMeter()

    # train
    if train:
        model.train()
        print("training...")
        for ep in range(epochs):
            tqdm_iter = tqdm(enumerate(train_data), mininterval=0.01, total=len(train_data))
            for i, (CT, NODULE) in tqdm_iter:
                nodule_idx = torch.where(NODULE != 0)[1]
                z_min, z_max = 0, CT.shape[1]-1
                last_z_idx = 0
                ct_list = []
                nodule_list = []
                for idx in nodule_idx:
                    if idx - 16 >= z_min and idx + 15 <= z_max and idx > last_z_idx:
                        ct = CT[0, idx-16:idx+16]
                        nodule = NODULE[0, idx-16:idx+16]
                        last_z_idx = idx+15
                        ct_list.append(ct)
                        nodule_list.append(nodule)
                # ct_list, nodule_list = gen_slices(CT, NODULE)
                if len(ct_list) and len(nodule_list):
                    ct = torch.stack(ct_list)
                    nodule = torch.stack(nodule_list)
                    if torch.cuda.is_available():
                        ct, nodule = ct.cuda(), nodule.cuda()
                    ct.unsqueeze_(dim=1)
                    nodule.unsqueeze_(dim=1)
                    optimizer.zero_grad()
                    feature, output = model(ct)
                    loss = criterion(output, nodule)
                    del ct, nodule
                    loss.backward()
                    optimizer.step()
                    train_meter.update(loss.item())
                    tqdm_iter.set_postfix(loss=f"{loss.item():.4f}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            train_meter.add_record()        
            torch.save(model.state_dict(), model_path)
            lr_scheduler.step()
        train_meter.plot(f"{UNI_SIGN}_train_loss.png")
        train_meter.plot(f"{UNI_SIGN}_unet_loss.txt")

    # eval
    model.eval()
    assert os.path.exists(model_path)
    print("evaluating...")
    I = 0
    with torch.no_grad():
        tqdm_iter = tqdm(enumerate(test_data), mininterval=0.01, total=len(test_data))
        for i, (CT, NODULE) in tqdm_iter:
            nodule_idx = torch.where(NODULE != 0)[1]
            z_min, z_max = 0, CT.shape[1]-1
            last_z_idx = 0
            ct_list = []
            nodule_list = []

            # make 32 slice with nodule
            for idx in nodule_idx:
                if idx - 16 >= z_min and idx + 15 <= z_max and idx > last_z_idx:
                    ct = CT[0, idx-16:idx+16]
                    nodule = NODULE[0, idx-16:idx+16]
                    last_z_idx = idx+15
                    ct_list.append(ct)
                    nodule_list.append(nodule)
            # ct_list, nodule_list = gen_slices(CT, NODULE)
            if len(ct_list) and len(nodule_list):
                ct = torch.stack(ct_list)
                nodule = torch.stack(nodule_list)
                ct, nodule = ct.unsqueeze(dim=1), nodule.unsqueeze(dim=1)
                if torch.cuda.is_available():
                    ct, nodule = ct.cuda(), nodule.cuda()
                feature, output = model(ct)
                loss = criterion(output, nodule.squeeze(dim=1))
                if I <= 15:
                    for _ct, _nodule, _output in zip(ct, nodule, output):
                        plot(_ct[0].cpu().squeeze(), _nodule[0].cpu().squeeze(), _output[0].cpu().squeeze(), img_save_path, I)
                        I += 1

                test_meter.update(loss.item())
                tqdm_iter.set_postfix(loss=f"{loss.item():.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    model = Unet3D()
    if torch.cuda.is_available():
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.6, weight_decay=1e-4)
    criterion = DiceLoss()
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1, 0.9)
    # criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter
    dir_path = os.path.dirname(os.path.abspath(__file__))
    img_save_path = os.path.join(project_path, "img")

    model_path = os.path.join(dir_path, f"{UNI_SIGN}_unet.pk")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))

    main(config["dst_ct_root"], config["dst_nodule_root"], train=False, epochs=200)