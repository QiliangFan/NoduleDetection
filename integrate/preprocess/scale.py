"""
使用Unet将原始数据的尺寸缩小到 1/4
因此label也需要缩小到1/4(向下取整)

Unet用的标签是: 结节标注

因此, 目标是生成 size/4 的结节标注

seriesuid,coordX,coordY,coordZ,diameter_mm
"""
import os
from glob import glob
from time import process_time

import math
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pandas.core.frame import DataFrame
from scipy.ndimage import zoom
from tqdm import tqdm
from multiprocessing import Pool

# 210
annotation_csv = "/home/fanrui/fanqiliang/lung16/CSVFILES/annotations.csv"
ct_mhd_root = "/home/fanrui/fanqiliang/lung16/LUNG16"
save_root = "/home/fanrui/fanqiliang/data/luna16/1_4_nodule"
raw_mhds = glob(os.path.join(ct_mhd_root, "**", "*.mhd"), recursive=True)
tq = tqdm(total=len(raw_mhds))


def work(mhd_path: str, nodule: DataFrame):
    save_path = mhd_path.replace(ct_mhd_root, save_root).replace(".mhd", ".npy")

    ct_img = sitk.ReadImage(mhd_path)
    ct_arr = sitk.GetArrayFromImage(ct_img)

    origin = ct_img.GetOrigin()  # (ox, oy, oz)
    space = ct_img.GetSpacing()  # (sx, sy, sz)
    direction = ct_img.GetDirection()  # (direc_x, direc_y, direc_z)

    nodule_arr: np.ndarray = np.zeros_like(ct_arr)
    for index, row in nodule.iterrows():
        cord_x = row["coordX"]
        cord_y = row["coordY"]
        cord_z = row["coordZ"]
        d = row["diameter_mm"]
        dx, dy, dz = d / space[0], d / space[1], d / space[2]
        rx, ry, rz = math.ceil(dx / 2), math.ceil(dy / 2), math.ceil(dz / 2)
        x, y, z = ct_img.TransformPhysicalPointToIndex((cord_x, cord_y, cord_z))

        min_x, max_x = max(0, x-rx), min(ct_arr.shape[2]-1, x+rx)
        min_y, max_y = max(0, y-ry), min(ct_arr.shape[1]-1, y+ry)
        min_z, max_z = max(0, z-rz), min(ct_arr.shape[0]-1, z+rz)

        nodule_arr[min_z:max_z, min_y:max_y, min_x:max_x] = 1
 
    # scale
    nodule_shape: np.ndarray = np.asarray(nodule_arr.shape)
    dst_nodule_shape: np.ndarray = np.floor(nodule_shape / 4)

    nodule_arr = zoom(nodule_arr, dst_nodule_shape / nodule_shape, mode="nearest")
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, nodule_arr)

    tq.update()


def main():
    # annotation
    annotation = pd.read_csv(annotation_csv)
    params = []
    for mhd in raw_mhds:
        sid = os.path.splitext(os.path.basename(mhd))[0]
        target = annotation[annotation["seriesuid"] == sid]
        params.append([mhd, target])
    with Pool(processes=2) as pool:
        pool.starmap(work, params)
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
