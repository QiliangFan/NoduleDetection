from skimage.segmentation import clear_border
from skimage.morphology import (binary_closing, binary_erosion,
                                convex_hull_image, disk)
from skimage.measure import label, regionprops
from skimage.filters import roberts
from scipy import ndimage
from numpy.core.fromnumeric import ndim
import numpy as np
import torch
from typing import Union
from multiprocessing import Pool


def seg_slice(slice: np.ndarray):
    if slice.ndim == 3 and slice.shape[0] == 1:
        slice = slice[0]
    assert slice.ndim == 2, f"expect arr to have ndim=2, but got ndim={slice.ndim}"
    arr = slice.copy()

    # binarize
    threshold = -600
    arr = arr < threshold

    # clear border
    arr = clear_border(arr)

    # divide into two areas (excluded with background)
    label_img = label(arr)
    areas = [r.area for r in regionprops(label_img)]
    areas.sort()
    labels = []
    if len(areas) > 2:
        for region in regionprops(label_img):
            if region.area < areas[-2]:
                for x, y in region.coords:
                    label_img[x, y] = 0
            else:
                x, y = region.coords[0]
                labels.append(label_img[x, y])
    else:
        labels = [1, 2]

    # fill holes
    l = label_img == labels[0]
    r = label_img == labels[1]
    l_edge = roberts(l)
    r_edge = roberts(r)
    l_region = ndimage.binary_fill_holes(l_edge)
    r_region = ndimage.binary_fill_holes(r_edge)

    # convex
    l = convex_hull_image(l)
    r = convex_hull_image(r)
    _ = label_img.copy()
    _[l == 1] = 1
    _[r == 1] = 1
    closing_img = binary_closing(_, selem=disk(10))

    result = slice * _


def seg(ct: Union[torch.Tensor, np.ndarray], use_multiprocess=True, processor=None) -> Uion[torch.Tensor, np.ndarray]:
    arr = ct.copy()

    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    if use_multiprocess:
        param_list = [slice for slice in arr]
        with Pool(processes=processor) as pool:
            result = pool.map(seg_slice, param_list)
            pool.close()
            pool.join()
        result = np.stack(result, axis=0)
    
    if isinstance(ct, torch.Tensor):
        result = torch.from_numpy(result)
    return result