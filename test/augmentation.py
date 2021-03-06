import os
from glob import glob
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import random
import time
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt

# dst_root = "/home/maling/fanqiliang/data/tmp"
dst_root = "/home/maling/fanqiliang/data/tmp64"

dst_ct_root = os.path.join(dst_root, "ct")
seg_root = "/home/maling/fanqiliang/lung16/seg-lungs-LUNA16"
candidate_file = "/home/maling/fanqiliang/lung16/CSVFILES/candidates_V2.csv"

# aug_patch_root = "/home/maling/fanqiliang/data/tmp/augmented_data"  # augmented data
aug_patch_root = "/home/maling/fanqiliang/data/tmp64/augmented_data"  # augmented data
if not os.path.exists(aug_patch_root):
    os.mkdir(aug_patch_root)

def work(sid, x, y, z, cls, I):
    if cls == 0:  # augmentation only for positive examples
        return
    random.seed(time.time())
    ct_file = glob(os.path.join(dst_ct_root, f"*/{sid}.mhd"))[0]
    img = sitk.ReadImage(ct_file)
    arr = sitk.GetArrayFromImage(img)
    output = arr.copy()
    shape = np.asarray(arr.shape)
    # z_r, y_r, x_r = 24, 24, 24
    z_r, y_r, x_r = 32, 32, 32
    i, j, k = img.TransformPhysicalPointToIndex([x, y, z])
    flip_op = [
        0,
        1,
        2,
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2)
    ]

    _K, _J, _I = k, j, i
    for z in range(10):  # copy them twenty times
        rand_z, rand_y, rand_x = random.randint(-4, 4), random.randint(-4, 4), random.randint(-4, 4)
        k = _K + rand_z
        j = _J + rand_y
        i = _I + rand_x

        # crop z
        min_z, max_z = np.clip(k-z_r, a_min=0, a_max=arr.shape[0]-1), \
                    np.clip(k+z_r, a_min=0, a_max=arr.shape[0]-1)
        output = arr[min_z:max_z]
        if output.shape[0] < 2*z_r:
            zero_pad = np.zeros((2*z_r-output.shape[0], output.shape[1], output.shape[2]))
            output = np.concatenate([output, zero_pad], axis=0)
        
        # crop y
        min_y, max_y = np.clip(j-y_r, a_min=0, a_max=arr.shape[1]-1), \
                    np.clip(j+y_r, a_min=0, a_max=arr.shape[1]-1)
        output = output[:, min_y:max_y]
        if output.shape[1] < 2*y_r:
            zero_pad = np.zeros((output.shape[0], 2*y_r-output.shape[1], output.shape[2]))
            output = np.concatenate([output, zero_pad], axis=1)
        
        # crop x
        min_x, max_x = np.clip(i-x_r, a_min=0, a_max=arr.shape[2]-1), \
                    np.clip(i+x_r, a_min=0, a_max=arr.shape[2]-1)
        output = output[:, :, min_x:max_x]
        if output.shape[2] < x_r*2:
            zero_pad = np.zeros((output.shape[0], output.shape[1], x_r*2-output.shape[2]))
            output = np.concatenate([output, zero_pad], axis=2)

        if max_z - min_z > z_r and max_y - min_y > y_r and max_x - min_x > x_r:
            np.save(os.path.join(aug_patch_root, f"{I}_{z}_7"), output.astype(np.float16))
            for k, flip_dim in enumerate(flip_op):
                _arr = np.flip(output, axis=flip_dim)

                np.save(os.path.join(aug_patch_root, f"{I}_{z}_{k}"), _arr.astype(np.float16))
        else:
            print("ERROR!", k, j, i)
        

def augmentation():
    candidate_csv = pd.read_csv(candidate_file)
    vals = candidate_csv.values

    task = []
    for i, row in tqdm(enumerate(vals), total=len(vals)):
        task.append([*row, i])
    with Pool(None) as pool:
        pool.starmap(work, task)
        print("prepare to return...")
        pool.close()
        pool.join()


if __name__ == "__main__":
    augmentation()