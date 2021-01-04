from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(project_path, "img")
patch_root = "/home/maling/fanqiliang/data/luna16/candidate"
files = glob(os.path.join(patch_root, "*.npy"))

for i in range(2):
    arr = np.load(files[i]).astype(np.float32)
    for j, data in enumerate(arr):
        plt.figure()
        plt.imshow(data, cmap="bone")
        plt.savefig(os.path.join(save_path, f"{i}_{j}.png"), bbox_inches="tight")
        plt.close()