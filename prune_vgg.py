'''Pruning VGG Model'''

import numpy as np
import os

import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load pre-trained model
net = VGG('VGG16')
net.to(device)

model_folder = 'trained_models'
filename = 'vgg16_trained.pth'
net.load_state_dict(torch.load(os.path.join(model_folder, filename)))