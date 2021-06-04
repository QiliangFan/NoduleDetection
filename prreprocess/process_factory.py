import os
from glob import glob
from multiprocessing import Pool
from prreprocess.utils import normalize, parenchyma_seg, unify_slice_thickness
from typing import Optional
from tqdm import tqdm
import SimpleITK as sitk

import pandas as pd
from config import config


def do(ct: str, dst_ct, nodule_csv: str, tq: tqdm):
    tq.update()
    
    ct: sitk.Image = sitk.ReadImage(ct)
    arr = sitk.GetArrayFromImage(ct)
    # unify thickness
    arr = unify_slice_thickness(arr, ct.GetSpacing())
    # segment parenchyma
    arr = parenchyma_seg(arr)
    # normalization
    arr = normalize(arr)

    

class ProcessFactory:
    def __init__(self, workers: Optional[int]=None):
        self.workers = workers

        # config
        self.nodule_csv: str = config.nodule_csv
        self.ct_root: str = config.ct_root
        self.output_path: str = config.output_path

    def work(self):
        ct_files = glob(os.path.join(self.ct_root, "**", "*.mhd"), recursive=True)
        dst_ct_files = [ct.replace(self.ct_root, os.path.join(self.output_path, "ct")) for ct in ct_files]
        tq = tqdm(total=len(ct_files))

        do_params = [(ct, dst_ct, self.nodule_csv, tq) for ct, dst_ct in zip(ct_files, dst_ct_files)]
        with Pool(processes=self.workers) as pool:
            pool.starmap(do, do_params)
            pool.close()
            pool.join()
        print("Work finished...")