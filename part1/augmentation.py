import os
import math
import random
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torchvision.transforms import ToPILImage
from torchvision.transforms import v2
from torchvision.io import read_image

plt.rcParams["savefig.bbox"] = 'tight'
to_image = ToPILImage()
torch.manual_seed(1)

renders = []
for render_set in os.listdir('./Renders/'):
    renders.append('./Renders/' + render_set)

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

# image reading
for i in range(len(renders)):
    img = read_image(str(Path(renders[i]) / 'render.jpg'))

img2 = to_image(blurrer(augmenter(img)))
imgplot = plt.imshow(img2)
plt.show()