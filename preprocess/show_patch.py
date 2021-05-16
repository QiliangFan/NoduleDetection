from glob import glob
import os
from os import path
import numpy as np
import matplotlib.pyplot as plt

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(project_path, "img")
if not os.path.exists(save_path):
    os.makedirs(save_path)
# patch_root = "/home/maling/fanqiliang/data/luna16/candidate"
# patch_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule/subset0"
# ct_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct/subset0"
# nodule_root = "/home/maling/fanqiliang/data/luna16/cube_nodule/subset0"
data_root = "/home/maling/fanqiliang/data/luna16/cube16_ct/subset0"
# aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"
# tmp_pos_root = "/home/maling/fanqiliang/data/tmp64/patch/1"
# tmp_neg_root = "/home/maling/fanqiliang/data/tmp64/patch/0"
# tmp_aug_root = "/home/maling/fanqiliang/data/tmp64/augmented_data"

files = glob(os.path.join(data_root, "*.npy"))
# files = ["/home/maling/fanqiliang/data/luna16/cube32_ct/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.898642529028521482602829374444_142.npy"]

for i in range(20):
    if i >= len(files): break
    arr = np.load(files[i]).astype(np.float32)
    # aug = np.load(files[i].replace(tmp_pos_root, tmp_aug_root)).astype(np.float32)
    if len(np.where(arr > 0)[0]) > 0:
        for j, data in enumerate(arr):
            plt.figure()
            plt.imshow(data, cmap="bone")
            plt.savefig(os.path.join(save_path, f"{i}_{j}.png"), bbox_inches="tight")
            plt.close()
        # for j, data in enumerate(aug):
        #     plt.figure()
        #     plt.imshow(data, cmap="bone")
        #     plt.savefig(os.path.join(save_path, f"aug_{i}_{j}.png"), bbox_inches="tight")
        #     plt.close()