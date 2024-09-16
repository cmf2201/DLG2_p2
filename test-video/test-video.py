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
from part2.network import *
from part2.utils import *
from dataloader import *

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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = VideoDataset(VIDEO_PATH)

model = UNET().to("cpu")
model.load_state_dict(torch.load("trained_model8.pth", map_location=torch.device('cpu')))

model.eval()
for batchcount, (rgb) in enumerate(dataset):
    rgb = rgb.to(device)
    
    pred = model(rgb)
    img = Image.fromarray(pred, 'L')
    img.save(str(batchcount + 1) + '.png')


# def val(dataloader, model, loss_fn, epochstep):
#     model.eval()
    
#     epochloss = 0
#     with torch.no_grad():
#         for batchcount, (rgb, label) in enumerate(dataloader):
#             dp(' batch', batchcount)
            
#             rgb = rgb.to(device)
#             label = label.to(device)
            
#             pred = model(rgb)
#             # print("Shapes: ")
#             # print(pred.shape)   
#             # print(label.shape)  
#             loss = loss_fn(pred, label)  

#             epochloss += loss.item()
        
#             wandb.log({
#                 "batch/loss/": loss.item(),
#                     })
            
#             if batchcount == 0: # only for the first batch every epoch
#                 wandb_images = []
#                 for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
#                     combined_image_np = CombineImages(pred_single, label_single, rgb_single)
#                     wandb_images.append(wandb.Image(combined_image_np))

#                 wandb.log(
#                 {
#                     "images/val": wandb_images,
#                 })
            
#     wandb.log({
#         "epoch/loss/val": epochloss,
#                 })