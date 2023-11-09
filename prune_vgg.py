'''Pruning VGG Model'''

import torch 
from Pruning_Engine import *

pruner = pruning_engine.PruningEngine(pruning_method='L1norm', individual=True)

from torchvision.models import vgg16_bn,VGG16_BN_Weights
weights = VGG16_BN_Weights.DEFAULT
model = vgg16_bn(weights=weights)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)