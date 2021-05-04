import numpy as np
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve

def main():
    csv = pd.read_csv("reduction/output.csv", header=None).values
    output = csv[:, 0]
    target = csv[:, 1].astype(np.int)

    precision, recall, threshold = precision_recall_curve(target, output)
    f1_score: np.ndarray = 2 * precision * recall / (precision + 10 * recall)
    arg_max = np.argmax(f1_score)
    print(f"f1-score: {f1_score[arg_max]}", f"precision: {precision[arg_max]}", f"recall: {recall[arg_max]}")

if __name__ == "__main__":
    main()