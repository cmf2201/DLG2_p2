#!/bin/bash

#SBATCH --mail-user=skushwaha@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=window_seg%j.out
#SBATCH --error=window_seg%j.err

#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -p academic
#SBATCH -t 23:00:00

module load miniconda3

# conda create --name "lab2"
source activate
conda activate project2

# conda install pytorch::pytorch
# conda install pytorch::torchvision
# conda install conda-forge::wandb
# conda install numpy
# conda install matplotlib

# module load cuda

python3 augmentation.py