import os
import shutil
import numpy as np
from PIL import Image


def dp(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)
            
def CombineImages(pred, label, rgb):
    pred = pred.detach().cpu().numpy().squeeze()
    label = label.detach().cpu().numpy().squeeze()
    rgb = rgb.detach().cpu().numpy()
    
    gray_array = 0.299 * rgb[0, :, :] + 0.587 * rgb[1, :, :] + 0.114 * rgb[2, :, :]

    # Concatenate images horizontally
    combined_image_np = np.concatenate((pred, label, gray_array), axis=1)
    combined_image_np = (np.clip(combined_image_np, 0, 1)*255).astype(np.uint8)
    return combined_image_np