from glob import glob
import os
import numpy as np
# import matplotlib
# matplotlib.use("agg")
# import matplotlib.pyplot as plt

aug_root = "/home/maling/fanqiliang/data/tmp/augmented_data"
files = glob(os.path.join(aug_root, "*"))

for i in range(10):
    file = files[i]
    for k in range(int(1000)):
        img = np.load(file)

        # plt.figure()
        # plt.imshow(slice, cmap="bone")
        # plt.savefig(f"img/{i}_{k}.png", bbox_inches="tight")
        # plt.axis("off")
        # plt.close()
