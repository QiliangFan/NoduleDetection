import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os


def show(ct_file):
    img = sitk.ReadImage(ct_file)
    arr = sitk.GetArrayFromImage(img)
    for i, slice in enumerate(arr):
        fig = plt.figure()
        plt.imshow(slice, cmap="bone")
        plt.axis("off")
        plt.savefig(f"img/{i}.png", bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    cube_ct = "/home/fanqiliang_be/data/luna16/cube_ct/subset0"
    cube_nodule = "/home/fanqiliang_be/data/luna16/cube_nodule/subset0"

    a1 = "1.3.6.1.4.1.14519.5.2.1.6279.6001.293757615532132808762625441831_0.mhd"
    show(os.path.join(cube_nodule, a1))
