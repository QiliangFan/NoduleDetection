from __future__ import print_function
from preprocess import scale_image, get_mhds, read_mhd, standardize
import os
from multiprocessing import Pool
import numpy as np
from datetime import datetime
import pandas as pd

scale_size = 0.5
dst_z_space = 1


def process_ct(mhd_file: str):
    file_name = os.path.splitext(os.path.basename(mhd_file))[0]
    img, arr = read_mhd(mhd_file)
    space = img.GetSpacing()
    arr.astype(np.float32)
    arr = scale_image(arr, scale=(space[2]/dst_z_space * scale_size, scale_size, scale_size))
    arr = standardize(arr)
    return arr, file_name


def process_label(save_dir, sid, mhd_file):
    img, arr = read_mhd(mhd_file)
    x_space, y_space, z_space = img.GetSpacing()

    arr = scale_image(arr, scale=(z_space/dst_z_space * scale_size, scale_size, scale_size))
    arr = np.rint(np.clip(arr, a_min=0, a_max=1)).astype(np.long)
    np.save(os.path.join(save_dir, sid), arr)
    print(f"{datetime.now():  %Y-%m-%d %H:%M:%S}", sid)


def process_nodule(dst_nodule: str, dst_size: str, dst_weight: str, sid, mhd_file, _x, _y, _z, _d):
    img, arr = read_mhd(mhd_file)
    x_space, y_space, z_space = img.GetSpacing()
    ox, oy, oz = img.GetOrigin()

    ix, iy, iz = (_x-ox) / x_space, (_y-oy) / y_space, (_z-oz) / dst_z_space
    ix, iy, iz = np.floor(ix * scale_size), np.floor(iy * scale_size), np.floor(iz * scale_size)

    dx, dy, dz = _d/x_space*scale_size, \
                 _d/y_space*scale_size, \
                 _d/dst_z_space*scale_size

    mask = np.zeros((int(arr.shape[0]*z_space/dst_z_space*scale_size),
                     int(arr.shape[1]*scale_size),
                     int(arr.shape[2]*scale_size)), dtype=np.long)
    sz = np.zeros((3,
                   int(arr.shape[0]*z_space/dst_z_space*scale_size),
                   int(arr.shape[1]*scale_size),
                   int(arr.shape[2]*scale_size)), dtype=np.float32)  # Chanel=3, each for x-y-z
    weight = np.zeros(mask.shape, dtype=np.float32)
    for k in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            for i in range(weight.shape[2]):
                MAX = 0
                for _ix, _iy, _iz, _dx, _dy, _dz in zip(ix, iy, iz, dx, dy, dz):
                    M = np.exp(-1 * ((i-_ix)**2/(2*_dx**2) + (j-_iy)**2/(2*_dy**2) + (k-_iz)**2/(2*_dz**2)) )
                    MAX = max(M, MAX)
                weight[k, j, i] = MAX
    for _ix, _iy, _iz, _dx, _dy, _dz in zip(ix, iy, iz, dx, dy, dz):
        rx, ry, rz = np.rint(_dx / 2), np.rint(_dy / 2), np.rint(_dz / 2)
        x_min, x_max = np.rint(np.clip(_ix-rx, a_min=0, a_max=mask.shape[2]-1)).astype(np.int), \
                       np.rint(np.clip(_ix+rx, a_min=0, a_max=mask.shape[2]-1)).astype(np.int)
        y_min, y_max = np.rint(np.clip(_iy-ry, a_min=0, a_max=mask.shape[1]-1)).astype(np.int), \
                       np.rint(np.clip(_iy+ry, a_min=0, a_max=mask.shape[1]-1)).astype(np.int)
        z_min, z_max = np.rint(np.clip(_iz-rz, a_min=0, a_max=mask.shape[0]-1)).astype(np.int), \
                       np.rint(np.clip(_iz+rz, a_min=0, a_max=mask.shape[0]-1)).astype(np.int)
        sz[0, z_min:z_max, y_min:y_max, x_min:x_max] = _dx
        sz[1, z_min:z_max, y_min:y_max, x_min:x_max] = _dy
        sz[2, z_min:z_max, y_min:y_max, x_min:x_max] = _dz
        mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
    np.save(os.path.join(dst_nodule, sid), mask)
    np.save(os.path.join(dst_size, sid), sz)
    np.save(os.path.join(dst_weight, sid), weight)
    print(f"{datetime.now(): nodule@@ %Y-%m-%d %H:%M:%S}", sid)


def gen_ct(ct_root: str, dst: str, workers=None):
    """
    multiprocessing supported
    :param ct_root:
    :param dst:
    :param workers:
    :return:
    """
    if not os.path.exists(ct_root):
        os.mkdir(ct_root)
    if not os.path.exists(dst):
        os.mkdir(dst)
    for subset in os.listdir(ct_root):
        subset_path = os.path.join(ct_root, subset)
        ct_list = get_mhds(subset_path)
        with Pool(processes=workers) as pool:
            result = pool.map(process_ct, ct_list)
            for arr, file_name in result:
                if not os.path.exists(os.path.join(dst, subset)):
                    os.mkdir(os.path.join(dst, subset))
                np.save(os.path.join(dst, subset, file_name), arr)
                print(f"{datetime.now(): ct@@ %Y-%m-%d %H:%M:%S}", file_name)


def gen_nodule(csv_file: str, ct_root: str, dst_nodule: str, dst_size: str, dst_weight: str, workers=None):
    assert os.path.exists(csv_file), "the csv file must exists!"
    if not os.path.exists(dst_nodule):
        os.mkdir(dst_nodule)
    if not os.path.exists(dst_size):
        os.mkdir(dst_size)
    if not os.path.exists(dst_weight):
        os.mkdir(dst_weight)
    csv = pd.read_csv(csv_file)
    cols = csv.columns
    sids = csv[cols[0]].to_numpy()
    x = csv[cols[1]].to_numpy()
    y = csv[cols[2]].to_numpy()
    z = csv[cols[3]].to_numpy()
    d = csv[cols[4]].to_numpy()

    nodule_tasks = []
    size_tasks = []
    for _ in os.listdir(ct_root):
        _ = os.path.join(ct_root, _)
        for file in os.listdir(_):
            if file.endswith(".mhd"):
                sid = os.path.splitext(file)[0]
                mhd_file = os.path.join(_, file)
                idx = np.where(sids == sid)
                _x = x[idx]
                _y = y[idx]
                _z = z[idx]
                _d = d[idx]

                nodule_tasks.append([dst_nodule, dst_size, dst_weight, sid, mhd_file, _x, _y, _z, _d])

    with Pool(processes=workers) as pool:
        pool.starmap(process_nodule, nodule_tasks)


def gen_seg(seg_root: str, dst_seg_root: str, workers=None):
    if not os.path.exists(dst_seg_root):
        os.mkdir(dst_seg_root)
    task = []
    for _ in os.listdir(seg_root):
        if _.endswith(".mhd"):
            sid = os.path.splitext(_)[0]
            file = os.path.join(seg_root, _)
            task.append([dst_seg_root, sid, file])

    with Pool(processes=workers) as pool:
        pool.starmap(process_label, task)















