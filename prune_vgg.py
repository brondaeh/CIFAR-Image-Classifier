'''Pruning VGG Model'''

import torch

from Models import *
from Pruner import *
from ptflops import get_model_complexity_info
from torchvision.models import vgg16

pruner = pruning_engine(pruning_method='L1norm', individual=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# from torchvision.models import vgg16_bn,VGG16_BN_Weights
# weights = VGG16_BN_Weights.DEFAULT
# model = vgg16_bn(weights=weights)
# model.to(device)

model = vgg16()
model.load_state_dict(torch.load('Trained_Models/vgg16_trained.pth', map_location=device))
model.to(device)

print ("--> Calculating Model Complexity Before Pruning...")

with torch.cuda.device(0):
    maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))