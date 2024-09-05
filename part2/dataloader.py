import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from dataaug import *
from loadParam import *
import pdb

class WindowDataset(Dataset):
    def __init__(self, ds_path):
        # init code
        print("dataset init")

    def __len__(self):
        # Set the dataset size here
        return N

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        
        # Open the RGB image and ground truth label

        # convert them to tensors

        # apply any transform (blur, noise...)
        
        return rgb, label


# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path=DS_PATH)
    dataloader = DataLoader(dataset)

    rgb, label = dataset[0]
