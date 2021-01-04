import json
from multiprocessing import Pool

from preprocess import Luna16Preprocess


def assign_task(boy, food):
    boy(*food)


def main():
    preprocess = Luna16Preprocess()
    preprocess.run()


if __name__ == "__main__":
    main()
