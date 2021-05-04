import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union

def show(arr: Union[np.ndarray, torch.Tensor]):
    if isinstance(arr, torch.Tensor):
        arr = np.asarray(arr)
    assert arr.ndim == 3
    for i, slice in enumerate(arr):
        # plt.imshow(np.expand_dims(slice, axis=0), cmap="bone")
        # plt.imshow(slice, cmap="bone")
        plt.imsave(f"img/{i}.png", slice, cmap="bone")
        plt.close()
    
        