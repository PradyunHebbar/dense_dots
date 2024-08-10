#!/bin/bash

module load anaconda/3/2023.03
module load pytorch/gpu-cuda-12.1/2.2.0 

python3 run.py --batch_size 64 --learning_rate 0.001 --num_epochs 1

# Eval
python3 plotting.py
