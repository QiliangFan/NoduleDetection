import os
from typing import Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import zoom
from skimage.filters import roberts
from skimage.measure import label, regionprops
from skimage.morphology import (binary_closing, binary_erosion,
                                convex_hull_image, disk)
from skimage.segmentation import clear_border
import pandas as pd

"""
Global Var
"""
DST_SLICE_THICKNESS = 1


def normalize(ct: Union[np.ndarray, str]) -> np.ndarray:
    if isinstance(ct, str):
        ct: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(ct))
    ct = (ct - ct.min()) / (ct.max() - ct.min())
    return ct


def unify_slice_thickness(ct: np.ndarray, space: Sequence) -> np.ndarray:
    """unify CT slice thickness to 1mm

    Args:
        ct (np.ndarray): intput data (Z, Y, X)
        space (np.ndarray): CT pixel space, (x, y, z)
    """
    space = np.asarray(space[::-1])
    dst_space = np.asarray((DST_SLICE_THICKNESS, space[1] * 2, space[2] * 2))
    factor = (space/dst_space).tolist()
    res = zoom(ct, zoom=factor, mode="nearest")
    return res


def parenchyma_seg(ct: np.ndarray) -> np.ndarray:
    res = np.zeros_like(ct)
    for i, s in enumerate(ct):
        arr = s.copy()

        # binarize
        threshold = -400
        arr = arr < threshold

        # clear border
        cleared = clear_border(arr)

        # divide into two areas (excluded with background)
        label_img = label(cleared)
        areas = [r.area for r in regionprops(label_img)]
        areas.sort()
        labels = []
        if len(areas) > 2:
            for region in regionprops(label_img):
                if region.area < areas[-2]:
                    for x, y in region.coords:
                        label_img[x, y] = 0
        arr = label_img > 0

        # fill holes
        arr = binary_erosion(arr, disk(2))
        arr = binary_closing(arr, disk(10))
        edges = roberts(arr)
        arr = ndimage.binary_fill_holes(edges)
        res[i] = arr * s

    return res


def gen_nodule(img: sitk.Image, nodule_csv: str, sid: str):
    import math
    nodule_csv: pd.DataFrame = pd.read_csv(nodule_csv)
    nodule_csv = nodule_csv[nodule_csv["seriesuid"] == sid]
    arr = sitk.GetArrayFromImage(img)
    nodule: np.ndarray = np.zeros_like(arr)
    for sid, x, y, z, d in nodule_csv.values:
        k, j, i = img.TransformPhysicalPointToIndex((x, y, z))
        space = img.GetSpacing()  
        space = space[::-1]  # (z, y, x)
        ri, rj, rk = math.ceil(d/space[0]/2), math.ceil(d/space[1]/2), math.ceil(d/space[2]/2)
        if 2 * ri * space[0] < 1: continue
        min_i, max_i = max(0, i-ri), min(i+ri, nodule.shape[0]-1)
        min_j, max_j = max(0, j-rj), min(j+rj, nodule.shape[1]-1)
        min_k, max_k = max(0, k-rk), min(k+rk, nodule.shape[2]-1)
        nodule[min_i:max_i, min_j:max_j, min_k:max_k] = 1
    return nodule