import os
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from tqdm import tqdm
import math
import SimpleITK as sitk


save_path = os.path.dirname(os.path.abspath(__file__))


def plot_z_size():
    sp = []
    files = glob(os.path.join(data_root, "*/*.mhd"), recursive=True)
    for f in tqdm(files):
        img = sitk.ReadImage(f)
        sp.append(img.GetSize()[2])
    plt.figure()
    _bin = list(range(min(sp), max(sp)+2))
    plt.title("Z size")
    plt.hist(sp, _bin)
    plt.xlabel("size")
    plt.ylabel("number")
    plt.savefig(os.path.join(save_path, "z_size.png"), bbox_inches="tight")
    plt.close()


def plot_space():
    x_space, y_space, z_space = [], [], []
    cts = glob(os.path.join(raw_data_root, "*/*.mhd"))
    for file in tqdm(cts):
        img = sitk.ReadImage(file)
        space = img.GetSpacing()
        x_space.append(space[0])
        y_space.append(space[1])
        z_space.append(space[2])
    data = np.stack([x_space, y_space, z_space], axis=0)
    np.savetxt(os.path.join(save_path, "space.txt"), data, fmt="%.2f")
    fig, ax = plt.subplots(3, 1)
    _bin = list(range(math.floor(data.min()), math.ceil(data.max()+1)))
    fig.suptitle("space")
    ax[0].set_title("x space")
    ax[0].hist(data[0])
    
    ax[1].set_title("y space")
    ax[1].hist(data[1])

    ax[2].set_title("z space")
    ax[2].hist(data[2])

    fig.tight_layout(pad=1.08, h_pad=1)
    plt.savefig(os.path.join(save_path, "space.png"), bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    raw_data_root = "/home/maling/fanqiliang/lung16/LUNG16"
    data_root = "/home/maling/fanqiliang/data/luna16/ct"

    # z size
    plot_z_size()

    # space
    # plot_space()
