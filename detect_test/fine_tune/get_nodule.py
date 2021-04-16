import os
import numpy as np
from glob import glob

ct_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
nodule_root = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"

nodule_list = glob(os.path.join(nodule_root, "**", "*.npy"))

for i, nodule in enumerate(nodule_list):
    nodule_arr = np.load(nodule).astype(np.float32)
    if len(np.where(nodule_arr > 0)[0]) > 1000:
        print(nodule, nodule.replace(nodule_root, ct_root))