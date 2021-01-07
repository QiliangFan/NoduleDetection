import numpy as np
import matplotlib.pyplot as plt
import os

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.nums = 0
        self.sum = 0 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.nums = 0
        self.sum = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.nums += n
        self.avg = self.sum / self.nums
    