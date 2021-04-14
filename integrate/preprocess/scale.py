"""
使用Unet将原始数据的尺寸缩小到 1/4
因此label也需要缩小到1/4(向下取整)

Unet用的标签是: 结节标注

因此, 目标是生成 size/4 的结节标注
"""
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom

# 210
annotation_csv = "/home/fanrui/fanqiliang/lung16/CSVFILES/annotations.csv"
ct_mhd_root = "/home/fanrui/fanqiliang/lung16/LUNG16"


def main():
    pass


if __name__ == "__main__":
    main()
