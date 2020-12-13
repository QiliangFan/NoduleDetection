import json
from multiprocessing import Pool

from preprocess.luna16 import gen_ct, gen_nodule, gen_seg


def assign_task(boy, food):
    boy(*food)


def main():
    # gen_ct(config["ct_root"], config["dst_ct_root"])
    # gen_nodule(config["csv_file"], config["ct_root"],
    #            config["dst_nodule_root"], config["dst_size_root"],
    #            config["weight_root"])
    gen_seg(config["seg_root"], config["dst_seg_root"])


if __name__ == "__main__":
    with open("config.json", "rb") as fp:
        config = json.load(fp)
    main()
