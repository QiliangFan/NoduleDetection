import os
from multiprocessing import Pool

import numpy as np
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import matplotlib as mpl

FPS = None
TPR = None


def get_roc_curve(label, score):
    label, score = label.reshape(-1), score.reshape(-1)
    num_neg = np.argwhere(label == 0).shape[0]
    fpr, tpr, thresholds = roc_curve(label, score)
    return fpr*num_neg, tpr, thresholds


def plot_froc_curve(fps: np.ndarray, tpr: np.ndarray):
    global FPS
    global TPR
    mpl.rcParams['agg.path.chunksize'] = 10000
    idx = np.argsort(fps)
    tpr = tpr[idx]
    fps = fps[idx]

    tick = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    idx = np.where(fps <= 8)[0]
    unique_fps = np.unique(fps[idx])
    unique_tpr = np.zeros_like(unique_fps)
    for i, _fps in enumerate(unique_fps):
        print(i, unique_fps.shape[0])
        _idx = np.where(fps == _fps)
        unique_tpr[i] = np.mean(tpr[_idx])

    plt.figure("FROC")
    plt.plot(unique_fps, unique_tpr)
    plt.xscale("log", base=2)
    plt.xlim([1/16, 10])
    plt.xticks(tick, tick)
    plt.savefig(os.path.join(project_path, "saving", "froc.png"))
    plt.close()


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fps = np.load(os.path.join(project_path, "saving", "fps.npy"))
    tpr = np.load(os.path.join(project_path, "saving", "tpr.npy"))
    plot_froc_curve(fps, tpr)