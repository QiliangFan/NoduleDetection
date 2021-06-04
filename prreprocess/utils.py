from typing import Sequence, Tuple, Union

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy import ndimage
from skimage.segmentation import clear_border
from skimage.morphology import (binary_closing, binary_erosion,
                                convex_hull_image, disk)
from skimage.measure import label, regionprops
from skimage.filters import roberts

"""
Global Var
"""
DST_SLICE_THICKNESS = 1


def normalize(ct: Union[np.ndarray, str]):
    if isinstance(ct, str):
        ct: np.ndarray = sitk.GetArrayFromImage(sitk.ReadImage(ct))
    ct = (ct - ct.min()) / (ct.max() - ct.min())
    return ct


def unify_slice_thickness(ct: np.ndarray, space: Sequence):
    """unify CT slice thickness to 1mm

    Args:
        ct (np.ndarray): intput data (Z, Y, X)
        space (np.ndarray): CT pixel space, (x, y, z)
    """
    space = np.asarray(space[::-1])
    dst_space = np.asarray((1, space[1], space[2]))
    factor = (space/dst_space).tolist()
    res = zoom(ct, zoom=factor, mode="nearest")
    return res
    
def parenchyma_seg(ct: np.ndarray):
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