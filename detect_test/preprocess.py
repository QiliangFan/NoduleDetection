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
if os.environ["IP"].endswith("210"):
    output_dir = "/home/fanrui/fanqiliang/data/luna16/cube_ct"
    input_dir = "/home/fanrui/fanqiliang/data/luna16/ct"
    output_nodule_dir = "/home/fanrui/fanqiliang/data/luna16/cube_nodule"
    annotation_csv = "/home/fanrui/fanqiliang/lung16/CSVFILES/annotations.csv"


# 219
if os.environ["IP"].endswith("219"):
    output_dir = "/home/fanqiliang_be/data/luna16/cube_ct"
    input_dir = "/home/fanqiliang_be/data/luna16/ct"
    output_nodule_dir = "/home/fanqiliang_be/data/luna16/cube_nodule"
    annotation_csv = "/home/fanqiliang_be/lung16/CSVFILES/annotations.csv"


def divide_with_stride(arr: np.ndarray) -> List[np.ndarray]:
    """
    (n, 256, 256) -> List[(64, 64, 64)]
    """

    result_list: List[np.ndarray] = []
    # slice by z axis
    for z in range(0, z_len := arr.shape[0], 64):
        if z + 63 >= z_len:
            z = z_len - 64
        z_arr: np.ndarray = arr[z:z+64]

        # slice by y axis
        for y in range(0, y_len := arr.shape[1], 64):
            y_arr: np.ndarray = z_arr[:, y:y+64]

            # slice by x axis
            for x in range(0, x_len := arr.shape[2], 64):
                x_arr: np.ndarray = y_arr[:, :, x:x+64]
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

    for idx, (_ct, _nodule) in enumerate(zip(ct_patchs, nodule_patches)):
        src_ct = dst_ct_file.replace(".mhd", f"_{idx}.npy")
        src_nodule = dst_nodule_file.replace(".mhd", f"_{idx}.npy")
        np.save(src_ct, _ct)
        np.save(src_nodule, _nodule)
        if len(np.where(_nodule > 0)[0]) > 0:
            augment(src_ct)


    # if shape[0] <= 256:  # 如果z space 小于等于256, 填0
    #     det_z = 256 - shape[0]
    #     arr = ct_data
    #     if det_z > 0:
    #         arr = np.concatenate(
    #             (arr, np.zeros((det_z, shape[1], shape[2]))), axis=0)
    #         nodule_data = np.concatenate(
    #             (nodule_data, np.zeros((det_z, shape[1], shape[2]))), axis=0)
    #     img = sitk.GetImageFromArray(arr)
    #     img.SetDirection(direction)
    #     img.SetSpacing(space)
    #     img.SetOrigin(origin)
    #     sitk.WriteImage(img, dst_ct_file)

    #     nodule = sitk.GetImageFromArray(nodule_data)
    #     nodule.SetDirection(direction)
    #     nodule.SetSpacing(space)
    #     nodule.SetOrigin(origin)
    #     sitk.WriteImage(nodule, dst_nodule_file)

    # else:   # 如果z space 大于等于256, 则划分成两个ct
    #     # 第一个ct
    #     arr1 = ct_data[:256, :, :]
    #     img1 = sitk.GetImageFromArray(arr1)
    #     img1.SetDirection(direction)
    #     img1.SetSpacing(space)
    #     img1.SetOrigin(origin)
    #     sitk.WriteImage(img1, dst_ct_file.replace(".mhd", "_0.mhd"))

    #     nodule_data1 = nodule_data[:256, :, :]
    #     nodule1 = sitk.GetImageFromArray(nodule_data1)
    #     nodule1.SetDirection(direction)
    #     nodule1.SetSpacing(space)
    #     nodule1.SetOrigin(origin)
    #     sitk.WriteImage(nodule1, dst_nodule_file.replace(".mhd", "_0.mhd"))

    #     # 第二个ct
    #     arr2 = ct_data[256:, :, :]
    #     nodule_data2 = nodule_data[256:, :, :]
    #     det_z = 256 - arr2.shape[0]
    #     if det_z >= 256:
    #         return
    #     elif det_z > 0:
    #         arr2 = np.concatenate(
    #             (arr2, np.zeros((det_z, arr2.shape[1], arr2.shape[2]))), axis=0)
    #         nodule_data2 = np.concatenate((nodule_data2, np.zeros(
    #             (det_z, arr2.shape[1], arr2.shape[2]))), axis=0)
    #     img2 = sitk.GetImageFromArray(arr2)
    #     img2.SetSpacing(space)
    #     img2.SetDirection(direction)
    #     # shape (x_shape, y_shape, z_shape)
    #     img1_shape = np.asarray(img1.GetSize())
    #     img2.SetOrigin(origin + img1_shape * space)
    #     sitk.WriteImage(img2, dst_ct_file.replace(".mhd", "_1.mhd"))

    #     nodule2 = sitk.GetImageFromArray(nodule_data2)
    #     nodule2.SetSpacing(space)
    #     nodule2.SetDirection(direction)
    #     nodule2.SetOrigin(origin + img1_shape * space)
    #     sitk.WriteImage(nodule2, dst_nodule_file.replace(".mhd", "_1.mhd"))


def main():

    with Pool(processes=8) as pool:
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
