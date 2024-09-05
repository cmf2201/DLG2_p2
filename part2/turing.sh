#!/bin/bash

#SBATCH --mail-user=username@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=/home/username/logs/window_seg%j.out
#SBATCH --error=/home/username/logs/window_seg%j.err

#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -C H100|A100|V100
#SBATCH -p short
#SBATCH -t 23:00:00

python3 train.py
