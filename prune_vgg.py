'''Pruning VGG Model'''

import torch

from Models import *
from Pruner import *
from ptflops import get_model_complexity_info
from torchvision.models import vgg16

'''
Load Pruning Method and Model
------------------------------------------------------
'''
pruner = pruning_engine(pruning_method='L1norm', individual=True)       # define L1norm pruning method
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
print(f'Device: {device}')

model = VGG('VGG16')
model.load_state_dict(torch.load('Trained_Models/vgg16_trained.pth', map_location=device))
model.to(device)

'''
Calculate Model Complexity Before Pruning
------------------------------------------------------
'''
print ("--> Calculating Model Complexity Before Pruning...")

with torch.cuda.device(0):
    maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))

'''
Prune the Model
------------------------------------------------------
'''
print ("--> Pruning VGG16 Model...")
pruned_layer = model.features[0]
pruner.set_pruning_ratio(0.1)
pruner.set_layer(pruned_layer,main_layer=True)
remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
model.features[0] = pruner.remove_filter_by_index(remove_filter_idx)

pruned_layer = model.features[1]
pruner.set_pruning_ratio(0.1)
pruner.set_layer(pruned_layer)
remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
model.features[1] = pruner.remove_Bn(remove_filter_idx)

pruned_layer = model.features[3]
pruner.set_pruning_ratio(0.1)
pruner.set_layer(pruned_layer)
remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
model.features[3] = pruner.remove_kernel_by_index(remove_filter_idx)

'''
Recalculate Model Complexity After Pruning
------------------------------------------------------
'''
print ("--> Calculating Model Complexity After Pruning...")

with torch.cuda.device(0):
    maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
    print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
