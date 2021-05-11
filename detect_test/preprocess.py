import os
from glob import glob
from math import ceil
from multiprocessing import Pool
from typing import List
import numpy as np
import pandas as pd
import SimpleITK as sitk

from augment import augment

# 210
# if os.environ["IP"].endswith("210"):
#     output_dir = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
#     input_dir = "/home/fanrui/fanqiliang/data/luna16/ct"
#     output_nodule_dir = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
#     annotation_csv = "/home/fanrui/fanqiliang/lung16/CSVFILES/annotations.csv"


# 219
# if os.environ["IP"].endswith("219"):
#     output_dir = "/home/fanqiliang_be/data/luna16/cube_ct"
#     input_dir = "/home/fanqiliang_be/data/luna16/ct"
#     output_nodule_dir = "/home/fanqiliang_be/data/luna16/cube_nodule"
#     annotation_csv = "/home/fanqiliang_be/lung16/CSVFILES/annotations.csv"

# 220
output_dir = "/home/maling/fanqiliang/data/luna16/cube16_ct"
input_dir = "/home/maling/fanqiliang/data/luna16/ct"
output_nodule_dir = "/home/maling/fanqiliang/data/luna16/cube16_nodule"
annotation_csv = "/home/maling/fanqiliang/lung16/CSVFILES/annotations.csv"


# 209



def divide_with_stride(arr: np.ndarray) -> List[np.ndarray]:
    """
    (n, 256, 256) -> List[(64, 64, 64)]
    """

    result_list: List[np.ndarray] = []
    # slice by z axis
    for z in range(0, z_len := arr.shape[0], 16):
        if z + 31 >= z_len:
            z = z_len - 16
        z_arr: np.ndarray = arr[z:z+16]

        # slice by y axis
        for y in range(0, y_len := arr.shape[1], 16):
            y_arr: np.ndarray = z_arr[:, y:y+16]

            # slice by x axis
            for x in range(0, x_len := arr.shape[2], 16):
                x_arr: np.ndarray = y_arr[:, :, x:x+16]
                if len(set(x_arr.shape)) == 1 and x_arr.shape[0] == 16:
                    result_list.append(x_arr)
                    print(x_arr.shape)
    return result_list


def work(ct_file: str) -> None:
    # nodule annotations
    nodule_pd = pd.read_csv(annotation_csv)
    sid = os.path.basename(ct_file).rstrip(".mhd")
    nodule_pd = nodule_pd[nodule_pd["seriesuid"].map(lambda x: sid in x)]
    print(f"{sid}: nodule number: {len(nodule_pd)} ")

    # filename to save with
    dst_ct_file = ct_file.replace(input_dir, output_dir)
    dst_nodule_file = ct_file.replace(input_dir, output_nodule_dir)
    if not os.path.exists(os.path.dirname(dst_ct_file)):
        os.makedirs(os.path.dirname(dst_ct_file), exist_ok=True)
    if not os.path.exists(os.path.dirname(dst_nodule_file)):
        os.makedirs(os.path.dirname(dst_nodule_file), exist_ok=True)

    ct_img = sitk.ReadImage(ct_file)
    space = np.asarray(ct_img.GetSpacing())  # space(x_sp, y_sp, z_sp)
    origin = np.asarray(ct_img.GetOrigin())  # (o_x, o_y, o_z)
    direction = ct_img.GetDirection()  # origin direction

    ct_data = sitk.GetArrayFromImage(ct_img)  # shape (z, y, x)
    shape = ct_data.shape  # shape(z_shape, y_shape, x_shape)

    # nodule data
    nodule_data = np.zeros_like(ct_data)
    for _, nodule in nodule_pd.iterrows():
        x = nodule["coordX"]
        y = nodule["coordY"]
        z = nodule["coordZ"]
        d = nodule["diameter_mm"]

        # z-i, y-j, x-k, d-(di, dj, dk)
        (k, j, i) = ct_img.TransformPhysicalPointToIndex((x, y, z))
        (dx, dy, dz) = np.ceil(np.divide(d, space))
        rx, ry, rz = ceil(dx/2), ceil(dy/2), ceil(dz/2)
        nodule_data[i-rz:i+rz, j-ry:j+ry, k-rx:k+rx] = 1

    # 切分为(64, 64, 64)的小方块
    ct_patchs = divide_with_stride(ct_data)
    nodule_patches = divide_with_stride(nodule_data)
    print("nums:", len(ct_patchs))

    for idx, (_ct, _nodule) in enumerate(zip(ct_patchs, nodule_patches)):
        src_ct = dst_ct_file.replace(".mhd", f"_{idx}.npy")
        src_nodule = dst_nodule_file.replace(".mhd", f"_{idx}.npy")
        np.save(src_ct, _ct)
        np.save(src_nodule, _nodule)
        if len(np.where(_nodule > 0)[0]) > 0:
            augment(src_ct)


def main():

    with Pool(processes=None) as pool:
        print("prepare to start...")
        pool.map(work, input_ct_list)
        pool.close()
        pool.join()
        print("prepare to exit...")

    # work(input_ct_list[0])


if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_ct_list = glob(os.path.join(
        input_dir, "**", "*.mhd"), recursive=True)
    print(f"预处理前列表长度: {len(input_ct_list)}")

    main()

    # 验证预处理结果
    result_ct_list = glob(os.path.join(
        output_dir, "**", "*.mhd"), recursive=True)
    print(f"预处理后列表长度: {len(result_ct_list)}")
