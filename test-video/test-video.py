# IMPORTS----------------------------------------------------------------------------
# STANDARD
import sys
import os
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torch.utils.data import Subset
# from torchmetrics.classification import Dice

import wandb

# CUSTOM
from network import *
from utils import *
from dataloader import *
import pdb
import utils

# Load the parameters
from loadParam import *
torch.cuda.empty_cache()

JOB_FOLDER = os.path.join(VIDEO_PATH, JOB_ID)
TRAINED_MDL_PATH = os.path.join(VIDEO_PATH, "trained_model")

# Ensure Outputs directory exists
if not os.path.exists(VIDEO_PATH):
    os.mkdir(VIDEO_PATH)

# Handle previous job folders if they exist
if os.path.exists(JOB_FOLDER):
    shutil.rmtree(JOB_FOLDER)
    print(f"Deleted previous job folder from {JOB_FOLDER}")

if os.path.exists(TRAINED_MDL_PATH):
    shutil.rmtree(TRAINED_MDL_PATH)
    print(f"Deleted previous training folder from {TRAINED_MDL_PATH}")


# Create necessary directories
os.mkdir(JOB_FOLDER)
os.mkdir(TRAINED_MDL_PATH)

datatype = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cuda:1')

dataset = VideoDataset(VIDEO_PATH)