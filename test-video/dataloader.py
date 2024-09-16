import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from torchvision.io import read_image
from PIL import Image
from pathlib import Path
import os
import glob
# from dataaug import *
from loadParam import *
import pdb
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, ds_path):
        self.frames = glob.glob(ds_path + '/test-video/*.jpg') # render folder path in relation to dataset path
        self.frames.sort(key=str.lower)

    def __len__(self):
        # Set the dataset size here
        return len(self.frames)

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        frames = self.frames[idx]
        
        # Open the RGB image and ground truth label
        # convert them to tensors

        frame = read_image(str(Path(frames)))

        # rgb = np.pad(rgb, ((0,0), (196,188), (196, 188)), mode='reflect').astype(np.float32)
        # label = np.pad(label, ((0,0), (196, 188), (196, 188)), mode='reflect').astype(np.float32)
        frame = frame / 255
        # apply any transform (blur, noise...)
        
        return frame