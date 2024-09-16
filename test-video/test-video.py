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

home_dir = "/home/caleb/Documents/WPI/24_25/DeepLearning/HW2/Group2_p2/test-video/"


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

    # # Move the tensor to the CPU, detach from computation graph, and convert to NumPy
    pred_np = pred.detach().numpy()

    #  # Optional: If the tensor has extra dimensions (batch or channel), you may need to squeeze
    # # Assuming pred is [batch_size, channels, height, width] and you want single-channel grayscale
    pred_np = np.squeeze(pred_np)  # Remove dimensions of size 1
    pred_np = np.squeeze(pred_np)  # Remove dimensions of size 1

    pred_np = np.clip(pred_np, 0, 1)

    # # Normalize the prediction to the range [0, 255] for image saving
    pred_np = (pred_np * 255).astype(np.uint8)

    # combined_image_np = CombineVideoImages(, pred)

    # img = Image.fromarray(combined_image_np, 'RGB')
    # img.save(os.path.join(home_dir,'Outputs/Combined/') + str(batchcount + 1) + '.png')

    img = Image.fromarray(pred_np, 'L')
    img.save(os.path.join(home_dir,'Outputs/Alone/') + str(batchcount + 1) + '.png')



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