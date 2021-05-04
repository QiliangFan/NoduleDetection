import os
from glob import glob
import numpy as np
import json

FOLD_NUM = 10

# 220
# pos_root = "/home/maling/fanqiliang/data/tmp/patch/1"
# neg_root = "/home/maling/fanqiliang/data/tmp/patch/0"
# aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"
pos_root = "/home/maling/fanqiliang/data/tmp64/patch/1"
neg_root = "/home/maling/fanqiliang/data/tmp64/patch/0"
aug_root = "/home/maling/fanqiliang/data/tmp64/augmented_data"

# 209
# pos_root = "/home/nku2/fanqiliang/data/tmp/patch/1"
# neg_root = "/home/nku2/fanqiliang/data/tmp/patch/0"
# aug_root = "/home/nku2/fanqiliang/data/tmp/augmented_data"

pos_files = glob(os.path.join(pos_root, "*.npy"))
neg_files = glob(os.path.join(neg_root, "*.npy"))
aug_files = glob(os.path.join(aug_root, "*.npy"))
pos_len = len(pos_files)
neg_len = len(neg_files)

DATA = {}

pos_data_list = []
neg_data_list = []

pos_idx = 0
neg_idx = 0
for i in range(10):
    pos_data_list.append(pos_files[pos_idx:pos_idx+pos_len//10])
    neg_data_list.append(neg_files[neg_idx:neg_idx+neg_len//10])
    pos_idx = pos_idx + pos_len//10
    neg_idx = neg_idx + neg_len//10

DATA["positive"] = pos_data_list
DATA["negative"] = neg_data_list
json.dump(DATA, open("data.json", "w"))