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

# Load the parameters
from loadParam import *
torch.cuda.empty_cache()

home_dir = "/Users/chaddyo/Desktop/coding projects/RBE474x/DLG2_p2/test-video/"


datatype = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = VideoDataset(os.path.join(home_dir,'images/'))

dataset = torch.utils.data.DataLoader(dataset, 1, False, num_workers=1)

model = nn.DataParallel(UNET()).to(device)

model.load_state_dict(torch.load(os.path.join(home_dir,"trained_model11.pth"), map_location=torch.device('cpu'),weights_only=True))
model.eval()

for batchcount, (rgb) in enumerate(dataset):
    rgb = rgb.to(device)
    
    pred = model(rgb)

    pred_np = pred.detach().numpy()

    pred_np = np.squeeze(pred_np)
    pred_np = np.squeeze(pred_np)

    pred_np = np.clip(pred_np, 0, 1)

    pred_np = (pred_np * 255).astype(np.uint8)

    # combined_image_np = CombineVideoImages(, pred)

    # img = Image.fromarray(combined_image_np, 'RGB')
    # img.save(os.path.join(home_dir,'Outputs/Combined/') + str(batchcount + 1) + '.png')

    img = Image.fromarray(pred_np, 'L')
    img.save(os.path.join(home_dir,'Outputs/Alone/') + str(batchcount + 1) + '.png')
