#!/bin/bash

#SBATCH --mail-user=skushwaha@wpi.edu
#SBATCH --mail-type=ALL

#SBATCH -J window_seg
#SBATCH --output=turing/window_seg%j.out
#SBATCH --error=turing/window_seg%j.err

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
conda activate lab2

# conda install pytorch::pytorch
# conda install pytorch::torchvision
# conda install conda-forge::wandb
# conda install numpy
# conda install matplotlib
# conda install <packages> -y

module load cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 train.py