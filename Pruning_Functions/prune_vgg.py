'''VGG Pruning Functions'''

import torch

from Models import *
from Pruner import *

def uniformPruneVGG16(model, pruning_ratio):
    '''
    Prunes every layer of the VGG16 model by the specified pruning ratio (excluding layers without parameters i.e. pooling and activation layers)

    Args:
    - model: the trained VGG16 model
    - pruning_ratio: the desired ratio of filters to prune

    Return: None
    '''
    print ("--> Uniformly pruning VGG16 model...")

    pruner = pruning_engine(pruning_method='L1norm', individual=True)       # define L1norm pruning method
    pruner.set_pruning_ratio(pruning_ratio)                                 # set pruning ratio

    layer_idx = 0
    for i, layer_cfg in enumerate(vgg.cfg['VGG16']):    # iterate over the cfg dictionary for VGG16
        if layer_cfg == 'M':    # skip to next element if maxpool
            continue
        else:
            pruned_layer = model.features[layer_idx]                                        # conv layer: prune filters
            pruner.set_layer(pruned_layer,main_layer=True)                                  # set pruned_layer in the pruner instance
            remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]             # obtain filter indices for pruning
            model.features[layer_idx] = pruner.remove_filter_by_index(remove_filter_idx)    # prune the filters in the layer

            layer_idx += 1  # increment layer_idx to next batchnorm layer
            pruned_layer = model.features[layer_idx]                                        # batchnorm layer: prune filters
            pruner.set_layer(pruned_layer)                                                  # set pruned_layer
            remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]             # obtain filter indices for pruning
            model.features[layer_idx] = pruner.remove_Bn(remove_filter_idx)                 # prune filters in the layer

            if i == 16: # if the end of cfg is reached, prune kernels of the final FC layer (classifier)
                pruned_layer = model.classifier                                             
                pruner.set_layer(pruned_layer)                             
                remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
                model.classifier = pruner.remove_kernel_by_index(remove_filter_idx, linear=True)
                break
            elif i < len(vgg.cfg['VGG16']) - 1: # else if the end of cfg is not reached yet, increment layer_idx if the next element of cfg is a maxpool layer
                next_layer_cfg = vgg.cfg['VGG16'][i + 1]

                if next_layer_cfg == 'M':
                    layer_idx += 1

            layer_idx += 2  # increment layer_idx to the next conv layer
            pruned_layer = model.features[layer_idx]                                        # next conv layer: prune kernels
            pruner.set_layer(pruned_layer)                                                  # set pruned_layer
            remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]             # obtain filter indices for pruning
            model.features[layer_idx] = pruner.remove_kernel_by_index(remove_filter_idx)    # prune kernels in the layer
