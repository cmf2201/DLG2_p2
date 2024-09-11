#!/bin/bash

#SBATCH --mail-user=ctnguyen@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=/home/ctnguyen/logs/window_seg%j.out
#SBATCH --error=/home/ctnguyen/logs/window_seg%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100
#SBATCH -p academic
#SBATCH -t 12:00:00

python3 train.py