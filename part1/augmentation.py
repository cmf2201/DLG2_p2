import os
import shutil
import math
import random
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torchvision.transforms.functional import perspective
from torchvision.transforms import ToPILImage
from torchvision.transforms import Pad
from torchvision.transforms import v2
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'
to_image = ToPILImage()
torch.manual_seed(1)

    ##                            ##
    ## FUNCTIONS FOR AUGMENTATION ##
    ##                            ##  

# Transformer
def perspective_transformer(img, alpha):
    
    endpoints = ((random.randint(0,250),random.randint(0,250)),
                 (0,2020-random.randint(0,250)),
                 (1180,2020-random.randint(0,250)),
                 (1180-random.randint(0,250),random.randint(0,250)))
    
    padder = Pad(100, fill=0, padding_mode='constant')
    perspective_render = perspective(img=padder(img), startpoints=((0,0),(0,2020),(1180,2020),(1180,0)),endpoints=endpoints)
    perspective_alpha = perspective(img=padder(alpha), startpoints=((0,0),(0,2020),(1180,2020),(1180,0)),endpoints=endpoints, fill=0)
    cropper = v2.CenterCrop(size=(int(1080/1.5),int(1920/1.5)))
    resizer = v2.Resize(size=(1080,1920))

    return (resizer(cropper(perspective_render)), resizer(cropper(perspective_alpha)))

# Blurrer
kernel_size = (random.randint(0, 10) * 2 + 9, random.randint(0, 10) * 2 + 15)
blurrer = v2.GaussianBlur(kernel_size=(21, 31), sigma=(0.1, 50.))
# Color Jitterer
brightnesser = v2.ColorJitter(brightness=random.random() / 2 + 0.1)
contraster = v2.ColorJitter(contrast=random.random() / 2 + 0.1)
saturationer = v2.ColorJitter(saturation=random.random() / 2 + 0.1)
huer = v2.ColorJitter(hue=random.random() / 2)
# Posterizer
bit_count = round(math.sqrt(random.randint(1,25)))
posterizer = v2.RandomPosterize(bits=bit_count, p=1)
# Solarizer
solarizer = v2.RandomSolarize(threshold=random.randint(128,255), p=1)
# Equalizer
equalizer = v2.RandomEqualize(p=1)
# Augmenter
augmenter = v2.AugMix(severity=3,mixture_width=1)

    ##                           ##
    ## TRANSFORM ALL RENDER SETS ##
    ##                           ##

# Image augmentation to entire folder
render_sets = []
for render_set in os.listdir('./Renders/'):
    render_sets.append('./Renders/' + render_set)

# Clear old and make new folder for transformed renders
if os.path.isdir('./TransformedRenders') : shutil.rmtree('./TransformedRenders')
os.mkdir('./TransformedRenders')

# transform and make new spots
for render_set in render_sets:
    render_folder_name = render_set.split("/")[-1]
    os.mkdir('./TransformedRenders/' + render_folder_name)
    render = read_image(str(Path(render_set) / 'render.jpg'))
    alpha = read_image(str(Path(render_set) / 'alpha_0000_0000.png'))
    
    per_render, per_alpha = perspective_transformer(render, alpha)
    image_render = to_image(per_render)
    image_alpha = to_image(per_alpha)

    image_render.convert('RGB')
    image_alpha.convert('L')

    render_path = "./TransformedRenders/" + render_folder_name + '/' +  os.path.basename('render.jpg')
    alpha_path = "./TransformedRenders/" + render_folder_name + '/' + os.path.basename('alpha_0000_0000.png')

    image_render.save(render_path)
    image_alpha.save(alpha_path)