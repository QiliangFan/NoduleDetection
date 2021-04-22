from glob import glob
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(project_path, "img")
# patch_root = "/home/maling/fanqiliang/data/luna16/candidate"
patch_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule/subset0"
ct_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct/subset0"

aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"
files = glob(os.path.join(aug_root, "*.npy"))

for i in range(100):
    arr = np.load(files[i]).astype(np.float32)
    ct = np.load(files[i].replace(patch_root, ct_root)).astype(np.float32)
    if len(np.where(arr > 0)[0]) > 0:
        for j, data in enumerate(arr):
            plt.figure()
            plt.imshow(data, cmap="bone")
            plt.savefig(os.path.join(save_path, f"{i}_{j}.png"), bbox_inches="tight")
            plt.close()
        for j, data in enumerate(ct):
            plt.figure()
            plt.imshow(data, cmap="bone")
            plt.savefig(os.path.join(save_path, f"ct_{i}_{j}.png"), bbox_inches="tight")
            plt.close()