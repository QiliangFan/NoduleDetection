import matplotlib
from numpy.core.fromnumeric import argsort, mean
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_curve

NUM_IMAGE = 888

def main():
    csv: np.ndarray  = pd.read_csv(sys.argv[1], header=None).values
    output = csv[:, 0]
    target = csv[:, 1]
    fpr, tpr, threshold = roc_curve(target, output)
    sort_arg = argsort(fpr)
    fpr = fpr[sort_arg]
    tpr = tpr[sort_arg]

    froc_fps = [1/8, 1/4, 1/2, 1, 2, 4, 8]
    froc_tpr = np.interp(froc_fps, fpr*NUM_IMAGE, tpr)
    print("froc tpr", froc_tpr)
    print("average tpr:", mean(froc_tpr))

    plt.figure()
    plt.title("FROC")
    for _fps, _tpr in zip(froc_fps, froc_tpr):
        plt.annotate(f"{_tpr:.4f}", (_fps , _tpr ), xytext=(_fps-0.2*_fps, _tpr-0.05))
    plt.xlabel("fps per scan")
    plt.ylabel("tpr")
    plt.ylim([0, 1])
    plt.xscale("log", base=2)
    plt.xticks(froc_fps)
    plt.plot(froc_fps, froc_tpr, color="violet")
    plt.fill_between(froc_fps, froc_tpr, 0, color="plum", alpha=0.4)
    plt.savefig("froc.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    assert len(sys.argv) == 2, f"expected two args but got {len(sys.argv)}"  # python3 froc.py output.file
    main()