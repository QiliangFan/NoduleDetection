from __future__ import print_function

import os

import scipy.ndimage as ndimage
import numpy as np
import SimpleITK as sitk


def scale_image(arr: np.ndarray, scale=1):
    return ndimage.zoom(arr, zoom=scale)


def get_mhds(dir: str):
    file_list = []
    files = os.listdir(dir)
    for _ in files:
        if _.endswith(".mhd"):
            file_list.append(os.path.join(dir, _))
    return file_list


def read_mhd(file):
    img = sitk.ReadImage(file)
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def standardize(arr):
    return (arr - np.min(arr)) / np.clip((np.max(arr) - np.min(arr)), a_min=1e-6, a_max=None)