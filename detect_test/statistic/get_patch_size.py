import os
from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import SimpleITK as sitk
from multiprocessing import Pool


def work(ct_file):
    ct_img = sitk.ReadImage(ct_file)
    ct_data = sitk.GetArrayFromImage(ct_img)
    return ct_data.shape

def main():
    x_shape = []
    y_shape = []
    z_shape = []

    with Pool(processes=None) as pool:
        result = pool.map(work, ct_list)
        pool.close()
        pool.join()
    for z, y, x in result:
        x_shape.append(x)
        y_shape.append(y)
        z_shape.append(z)
    
    x_shape = np.asarray(x_shape)
    y_shape = np.asarray(y_shape)
    z_shape = np.asarray(z_shape)

    plt.figure()
    plt.title("Size statistic...")
    plt.hist([x_shape, y_shape, z_shape], bins=100, label=["x_shape", "y_shape", "z_shape"])
    plt.legend()
    plt.savefig("size.png", bbox_inches="tight")


if __name__ == "__main__":
    processed_ct_root = "/home/fanqiliang_be/data/luna16/ct"
    ct_list = glob(os.path.join(processed_ct_root, "**", "*.mhd"), recursive=True)
    main()