import numpy as np
import matplotlib.pyplot as plt
import os

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.val_list = []
        self.avg = 0
        self.nums = 0
        self.sum = 0 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.nums = 0
        self.sum = 0

    def clear_record(self):
        self.val_list.clear()
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.nums += n
        self.avg = self.sum / self.nums
    
    def add_record(self, val=None):
        if val is None:
            self.val_list.append(self.avg)
        else:
            self.val_list.append(val)

    def plot(self, file):
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        plt.figure()
        plt.plot(self.val_list)
        plt.savefig(os.path.join(project_path, "result", file), bbox_inches="tight")
        plt.xlim([min(self.val_list)-abs(min(self.val_list))/10, max(self.val_list)*11/10])
        plt.close()

    def save(self, file):
        project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(project_path, "result", file), "a") as fp:
            print(*map(lambda x: str(x), self.val_list), sep="\n", file=fp)