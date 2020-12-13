import os

import torch
import torch.nn as nn
import json
from multi_task import MultiTask


def main():
    multi_task.cross_validation(test=False, evaluate=False)


if __name__ == "__main__":
    project_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(project_path, "config.json")
    with open(config_file, "rb") as fp:
        config = json.load(fp)
    multi_task = MultiTask(config["dst_ct_root"],
                           config["dst_seg_root"],
                           config["dst_nodule_root"],
                           config["dst_size_root"],
                           config["weight_root"]
                           )
    main()
