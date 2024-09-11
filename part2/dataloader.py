import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from PIL import Image
from pathlib import Path
import os
import glob
# from dataaug import *
from loadParam import *
import pdb

class WindowDataset(Dataset):
    def __init__(self, ds_path):
        # init code
        print("dataset init")
        self.renders = glob.glob(ds_path + '/Renders/*.*') # render folder path in relation to dataset path
        self.labels = glob.glob(ds_path + '/Labels/*.*') # alpha folder path in relation to dataset path
        self.renders.sort(key=str.lower)
        self.labels.sort(key=str.lower)
        

    def __len__(self):
        # Set the dataset size here
        return len(self.labels)

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        render = self.renders[idx]
        alpha = self.labels[idx]
        
        # Open the RGB image and ground truth label
        # convert them to tensors

        rgb = read_image(str(Path(render)))
        label = read_image(str(Path(alpha)))

        # apply any transform (blur, noise...)
        
        return rgb, label


# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path=DS_PATH)
    dataloader = DataLoader(dataset)

    rgb, label = dataset[0]