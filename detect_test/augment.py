import numpy as np
import os
from glob import glob
import shutil

# 210
aug_root = "/home/fanrui/fanqiliang/data/luna16/cube_aug"
ct_root = "/home/fanrui/fanqiliang/data/luna16/cube_ct"


def augment(src_file: str):
    if not os.path.exists(src_file):
        print(f"{src_file} does not exist...")
    dst_file = src_file.replace(ct_root, aug_root)
    if not os.path.exists(os.path.dirname(dst_file)):
        os.makedirs(os.path.dirname(dst_file))
    shutil.copyfile(src_file, dst_file)
