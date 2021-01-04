"""
1. crop to pulmary parenchyma.
2. only make use of the parenchyma region. mask off the rest regions.
3. generate some patches to get candidate nodules

space and origin information must be stored!
"""

import json
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from scipy import ndimage
from tqdm import tqdm


"""
  "ct_root"
  "csv_file"
  "sid_file"
  "seg_root"
  "dst_root"
  "dst_ct_root"
  "dst_seg_ct_root"
  "dst_nodule_root"
  "dst_size_root"
  "weight_root"
  "dst_seg_root"
"""
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
config_file = json.load(open(config_path, "rb"))

def ct_worker(file, seg_file, save_path):
    """
    :param binary_value: If True, returnValue can only contains {0, 1}
    """
    img = sitk.ReadImage(file)
    arr = sitk.GetArrayFromImage(img).astype(np.float16)
    seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
    arr = arr*seg
    arr = np.clip(arr, a_min=-1200, a_max=600)  # clip to [-1200, 600]

    # ct information
    origin = img.GetOrigin()
    space = img.GetSpacing()

    # target ct information
    target_origin = origin
    target_space = (space[0]*2, space[1]*2, 1)

    # unify the z space and half scale xy
    scale = (space[2]/target_space[2], space[1]/target_space[1], space[0]/target_space[0])
    target_arr = ndimage.zoom(arr, scale)
    target_arr = (target_arr - target_arr.min()) / (target_arr.max() - target_arr.min())

    # zero_pad = np.zeros((256-target_arr.shape[0], target_arr.shape[1], target_arr.shape[2]), dtype=np.float16)
    # target_arr = np.concatenate([target_arr, zero_pad], axis=0)
    target_img = sitk.GetImageFromArray(target_arr)
    target_img.SetDirection(img.GetDirection())
    target_img.SetOrigin(target_origin)
    target_img.SetSpacing(target_space)

    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    sitk.WriteImage(target_img, save_path)
    print(f"{datetime.now(): %Y-%m-%d %H:%M:%S}", save_path)

def nodule_worker(centers, diameters, ct_file, save_path):
    img = sitk.ReadImage(ct_file)
    arr = sitk.GetArrayFromImage(img)
    origin = img.GetOrigin()
    space = img.GetSpacing()
    sz = img.GetSize()

    mask = np.zeros(arr.shape, dtype=np.int16)
    for (x, y, z), d in zip(centers, diameters):
        i, j , k = np.round((x-origin[0])/space[0]).astype(np.int), \
                   np.round((y-origin[1])/space[1]).astype(np.int), \
                   np.round((z-origin[2])/space[2]).astype(np.int)
        rx, ry, rz = np.round(d/space[0]/2).astype(np.int), \
                     np.round(d/space[1]/2).astype(np.int), \
                     np.round(d/space[2]/2).astype(np.int)

        min_i, max_i = np.clip(i-rx, a_min=0, a_max=sz[0]-1), \
                       np.clip(i+rx, a_min=0, a_max=sz[0]-1)
        min_j, max_j = np.clip(j-ry, a_min=0, a_max=sz[1]-1), \
                       np.clip(j+ry, a_min=0, a_max=sz[1]-1)
        min_k, max_k = np.clip(k-rz, a_min=0, a_max=sz[2]-1), \
                       np.clip(k+rz, a_min=0, a_max=sz[2]-1)
        mask[min_k:max_k, min_j:max_j, min_i:max_i] = 1
    target_img = sitk.GetImageFromArray(mask)
    target_img.SetDirection(img.GetDirection())
    target_img.SetOrigin(origin=origin)
    target_img.SetSpacing(space)

    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    sitk.WriteImage(target_img, save_path)
    print(f"{datetime.now(): %Y-%m-%d %H:%M:%S}", save_path)

class Luna16Preprocess:
    def __init__(self):
        self.ct_root = config_file["ct_root"]
        self.csv_file = config_file["csv_file"]
        self.sid_file = config_file["sid_file"]
        self.seg_root = config_file["seg_root"]
        self.dst_root = config_file["dst_root"]
        self.dst_ct_root = config_file["dst_ct_root"]
        self.dst_seg_ct = config_file["dst_seg_ct_root"]
        self.dst_nodule_root = config_file["dst_nodule_root"]
        self.dst_size_root = config_file["dst_size_root"]
        self.weight_root = config_file["weight_root"]
        self.dst_seg_root = config_file["dst_seg_root"]

        # create directory
        self.__confirm_exists()

    def run(self, workers=None):
        print(f"{datetime.now(): %Y-%m-%d %H:%M:%S} start ct process...")
        self.gen_ct(self.ct_root, self.dst_ct_root, self.seg_root, workers=workers)
        print(f"{datetime.now(): %Y-%m-%d %H:%M:%S} start nodule process...")
        self.gen_nodule(self.csv_file, self.dst_nodule_root, workers=workers)

    def gen_ct(self, ct_root: str, dst: str,seg_root: str, workers=None):
        """
        multiprocessing supported method to generate ct data
        :param ct_root:
        :param dst:
        :param workers:
        :return:
        """
        src_list = glob(os.path.join(ct_root, "*/*.mhd"))
        seg_list = [os.path.join(seg_root, os.path.basename(_)) for _ in src_list]
        dst_list = [_.replace(ct_root, dst) for _ in src_list]
        with Pool(processes=workers) as pool:
            pool.starmap(ct_worker, list(zip(src_list, seg_list, dst_list)))
            pool.close()
            print("to be continue...")
            pool.join()

    def gen_nodule(self, csv_file, dst: str, workers=None):
        """
        :param csv_file: nodule annotation csv
        :param dst: 
        :return:
        """
        sid_csv = pd.read_csv(self.sid_file, header=None)
        total_sids = sid_csv[sid_csv.columns[0]]
        csv = pd.read_csv(csv_file)
        sids, xs, ys, zs, ds = csv.values.transpose()
        del sid_csv
        tasks = []
        for i, sid in tqdm(enumerate(total_sids)):
            idx = np.where(sids == sid)
            if idx[0].size:
                centers = np.stack((xs[idx], ys[idx], zs[idx]), axis=1)
                diameters = ds[idx]
            else:
                centers = []
                diameters = []
            ct_file = glob(os.path.join(self.dst_ct_root, f"*/{sid}.mhd"))[0]
            save_path = ct_file.replace(self.dst_ct_root, dst)
            tasks.append((centers, diameters, ct_file, save_path))
        with Pool(processes=workers) as pool:
            pool.starmap(nodule_worker, tasks)
            pool.close()
            print("to be continue...")
            pool.join()

    def __confirm_exists(self):
        if not os.path.exists(self.dst_root):
            os.mkdir(self.dst_root)
        if not os.path.exists(self.dst_ct_root):
            os.mkdir(self.dst_ct_root)
        if not os.path.exists(self.dst_seg_ct):
            os.mkdir(self.dst_seg_ct)
        if not os.path.exists(self.dst_nodule_root):
            os.mkdir(self.dst_nodule_root)
        if not os.path.exists(self.dst_size_root):
            os.mkdir(self.dst_size_root)
        if not os.path.exists(self.weight_root):
            os.mkdir(self.weight_root)
        if not os.path.exists(self.dst_seg_root):
            os.mkdir(self.dst_seg_root)
