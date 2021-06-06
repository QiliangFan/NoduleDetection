import os
from glob import glob
from multiprocessing import Pool
from preprocess.utils import gen_nodule, normalize, parenchyma_seg, unify_slice_thickness
from typing import Optional
import json
import SimpleITK as sitk
import numpy as np

import pandas as pd
from config import config

# global variable
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def do(ct: str, dst_ct: str, dst_nodule: str, nodule_csv: str):
    sid = os.path.splitext(os.path.basename(ct))[0]
    img: sitk.Image = sitk.ReadImage(ct)
    nodule = gen_nodule(img, nodule_csv, sid)
    arr = sitk.GetArrayFromImage(img)
    # unify thickness
    arr = unify_slice_thickness(arr, img.GetSpacing())
    nodule = unify_slice_thickness(nodule, img.GetSpacing())
    assert arr.shape[0] == nodule.shape[0]
    # segment parenchyma
    arr = parenchyma_seg(arr)
    # normalization
    arr = normalize(arr)

    # arr_save = dst_ct.replace(".mhd", ".npy")
    # nodule_save = dst_nodule.replace(".mhd", ".npy")
    # if not os.path.exists(os.path.dirname(arr_save)):
    #     os.makedirs(os.path.dirname(arr_save), exist_ok=True)
    # if not os.path.exists(os.path.dirname(nodule_save)):
    #     os.makedirs(os.path.dirname(nodule_save), exist_ok=True)
    # np.save(arr_save, arr)
    # np.save(nodule_save, nodule)

    print(sid, arr.shape)


    # divide slice
    for i in range(0, arr.shape[0], 64):
        if i + 64 >= arr.shape[0]+1: continue
        arr_slice = arr[i:i+64]
        nodule_slice = nodule[i:i+64]
        assert nodule_slice.shape[0] == 64 and arr_slice.shape[0] == 64
        arr_save = dst_ct.replace(".mhd", f"_{i}.npy")
        nodule_save = dst_nodule.replace(".mhd", f"_{i}.npy")
        bbox_save = nodule_save.replace("nodule", "bbox").replace(".npy", ".json")
        if not os.path.exists(os.path.dirname(arr_save)):
            os.makedirs(os.path.dirname(arr_save), exist_ok=True)
        if not os.path.exists(os.path.dirname(nodule_save)):
            os.makedirs(os.path.dirname(nodule_save), exist_ok=True)
        if not os.path.exists(os.path.dirname(bbox_save)):
            os.makedirs(os.path.dirname(bbox_save), exist_ok=True)

        from skimage.measure import label, regionprops
        lable_nodule = label(nodule_slice)
        bboxs = [r.bbox for r in regionprops(lable_nodule)]
        if len(bboxs) > 0:
            json.dump(bboxs, open(bbox_save, "w"))
        np.save(arr_save, arr_slice)
        np.save(nodule_save, nodule_slice)
    

class ProcessFactory:
    def __init__(self, workers: Optional[int]=None):
        self.workers = workers

        # config
        self.nodule_csv: str = config["nodule_csv"]
        self.ct_root: str = config["ct_root"]
        self.output_path: str = config["output_path"]

    def work(self):
        ct_files = glob(os.path.join(self.ct_root, "**", "*.mhd"), recursive=True)
        dst_ct_files = [ct.replace(self.ct_root, os.path.join(self.output_path, "ct")) for ct in ct_files]
        dst_nodule_files = [ct.replace(self.ct_root, os.path.join(self.output_path, "nodule")) for ct in ct_files]

        do_params = [(ct, dst_ct, dst_nodule, self.nodule_csv) for ct, dst_ct, dst_nodule in zip(ct_files, dst_ct_files, dst_nodule_files)]
        with Pool(processes=self.workers) as pool:
            pool.starmap(do, do_params)
            pool.close()
            pool.join()
        print("Work finished...")