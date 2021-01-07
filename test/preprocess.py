import os
from datetime import datetime
import numpy as np
from scipy import ndimage
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import pandas as pd

# path
dst_root = "/home/maling/fanqiliang/data/tmp"
dst_ct_root = os.path.join(dst_root, "ct")
patch_root = "/home/maling/fanqiliang/data/tmp/patch"   # patched root 
aug_patch_root = "/home/maling/fanqiliang/data/tmp/augmented_data"  # augmented data

if not os.path.exists(dst_ct_root):
    os.mkdir(dst_ct_root)
if not os.path.exists(patch_root):
    os.mkdir(patch_root)
if not os.path.exists(aug_patch_root):
    os.mkdir(aug_patch_root)

ct_root = "/home/maling/fanqiliang/lung16/LUNG16"
seg_root = "/home/maling/fanqiliang/lung16/seg-lungs-LUNA16"
candidate_file = "/home/maling/fanqiliang/lung16/CSVFILES/candidates_V2.csv"


def resample(image, spacing, new_spacing=(1, 1, 1)):
    spacing = np.asarray(spacing)
    new_spacing = np.asarray(new_spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_facotr = new_shape / np.asarray(image.shape)
    new_spacing = spacing * image.shape / new_shape
    image = ndimage.zoom(image, real_resize_facotr, mode="nearest")
    new_spacing = np.flip(new_spacing, axis=0)
    return image, new_spacing, real_resize_facotr

def make_ct(ct_file, seg_file, save_path):
    img = sitk.ReadImage(ct_file)
    arr = sitk.GetArrayFromImage(img).astype(np.float16)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
    arr = arr*seg
    arr = np.clip(arr, a_min=-1200, a_max=600)  # clip to [-1200, 600]

    # ct information
    origin = img.GetOrigin()
    space = np.asarray(img.GetSpacing())
    space = np.flip(space, axis=0)

    arr, target_space, resize_factor = resample(arr, space)
    arr = (arr - arr.min()) / (arr.max() - arr.min())

    target_img = sitk.GetImageFromArray(arr)
    target_img.SetDirection(img.GetDirection())
    target_img.SetOrigin(origin)
    target_img.SetSpacing(target_space)

    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    sitk.WriteImage(target_img, save_path)
    print(f"{datetime.now(): %Y-%m-%d %H:%M:%S}", save_path)

def candidate_worker(sid, x, y, z, cls, I):
    """
    (24, 40, 40)
    """
    z_r, y_r, x_r = 12, 20, 20
    
    class_dir = os.path.join(patch_root, f"{cls}")
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)

    file_name = os.path.join(class_dir, f"{I}.npy")
    img = sitk.ReadImage(glob(os.path.join(dst_ct_root, f"subset*/{sid}.mhd"))[0])
    arr = sitk.GetArrayFromImage(img)
    i, j, k = img.TransformPhysicalPointToIndex([x, y, z])
    output = arr
    # crop z
    min_z, max_z = np.clip(k-z_r, a_min=0, a_max=arr.shape[0]-1), \
                   np.clip(k+z_r, a_min=0, a_max=arr.shape[0]-1)
    output = output[min_z:max_z]
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

    assert output.shape[0] == 24 and \
           output.shape[1] == 40 and \
           output.shape[2] == 40, "expected shape (24, 40, 40) !"
    output = output.astype(np.float16)
    np.save(file_name, output)
    print(file_name)

def ct_work():
    # make ct
    def do(ct_path: str, seg_dir: str, dst_dir: str):
        seg_file = os.path.join(seg_dir, os.path.basename(ct_path))
        dst_file = ct_path.replace(ct_root, dst_dir)
        if not os.path.dirname(dst_file):
            os.mkdir(os.path.dirname(dst_file))
        return ct_path, seg_file, dst_file
    ct_task = [do(ct_path, seg_root, dst_ct_root) for ct_path in glob(os.path.join(ct_root, "*/*.mhd"))]
    print("begin to process ct")
    with Pool(processes=None) as pool:
        pool.starmap(make_ct, ct_task)
        pool.close()
        pool.join()
        print("ct task is done!")

def patch_work():
    # make patch
    candidate_csv = pd.read_csv(candidate_file)
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
    ct_work()

    patch_work() 