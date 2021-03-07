#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 test/train.py --network=base --epochs=2 --writer=True
