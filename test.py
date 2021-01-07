import SimpleITK as sitk
from glob import glob
import os

ct_root = "/home/maling/fanqiliang/data/tmp/ct"
ct_file = glob(os.path.join(ct_root, "*/*.mhd"))[200]
img = sitk.ReadImage(ct_file)
print(img.GetSpacing())