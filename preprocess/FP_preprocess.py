"""
FP reduction_generate many patchs
All patch generate in order, filename is relevant to the index of rows
"""
import json
import os
import pandas as pd
import SimpleITK as sitk
from glob import glob
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm


project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_file = os.path.join(project_path, "config.json")
config = json.load(open(config_file, "rb"))

ct_path = config["dst_ct_root"]
candidate_file = config["candidate_file"]
output_path = config["candidate_patch"]
PATCH_SIZE = config["patch_size"]


def candidate_worker(sid, x, y, z, cls, I):
    file_name = os.path.join(output_path, f"{I}.npy")
    img = sitk.ReadImage(glob(os.path.join(ct_path, f"subset*/{sid}.mhd"))[0])
    arr = sitk.GetArrayFromImage(img)
    i, j, k = img.TransformPhysicalPointToIndex([x, y, z])
    output = arr
    # crop z
    min_z, max_z = np.clip(k-PATCH_SIZE//2, a_min=0, a_max=arr.shape[0]-1), \
                   np.clip(k+PATCH_SIZE//2, a_min=0, a_max=arr.shape[0]-1)
    output = output[min_z:max_z]
    if output.shape[0] < PATCH_SIZE:
        zero_pad = np.zeros((PATCH_SIZE-output.shape[0], output.shape[1], output.shape[2]))
        output = np.concatenate([output, zero_pad], axis=0)
    
    # crop y
    min_y, max_y = np.clip(j-PATCH_SIZE//2, a_min=0, a_max=arr.shape[1]-1), \
                   np.clip(j+PATCH_SIZE//2, a_min=0, a_max=arr.shape[1]-1)
    output = output[:, min_y:max_y]
    if output.shape[1] < PATCH_SIZE:
        zero_pad = np.zeros((output.shape[0], PATCH_SIZE-output.shape[1], output.shape[2]))
        output = np.concatenate([output, zero_pad], axis=1)
    
    # crop x
    min_x, max_x = np.clip(i-PATCH_SIZE//2, a_min=0, a_max=arr.shape[2]-1), \
                   np.clip(i+PATCH_SIZE//2, a_min=0, a_max=arr.shape[2]-1)
    output = output[:, :, min_x:max_x]
    if output.shape[2] < PATCH_SIZE:
        zero_pad = np.zeros((output.shape[0], output.shape[1], PATCH_SIZE-output.shape[2]))
        output = np.concatenate([output, zero_pad], axis=2)

    assert len(set(output.shape)) == 1 and PATCH_SIZE in set(output.shape), f"{PATCH_SIZE} is expected, but got {output.shape}"
    output = output.astype(np.float16)
    np.save(file_name, output)
    print(file_name)


def gen_patch():
    candidate_csv = pd.read_csv(candidate_file)
    cols = candidate_csv.columns 
    vals = candidate_csv.values

    task = []
    for i, row in tqdm(enumerate(vals), total=len(vals)):
        task.append([*row, i])
    with Pool(None) as pool:
        pool.starmap(candidate_worker, task)
        print("prepare to return...")
        pool.close()
        pool.join()
    
if __name__ == "__main__":
    gen_patch()
