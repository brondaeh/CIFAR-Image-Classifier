'''Pruning Engine Base Class in PyTorch'''

import torch

class pruning_engine_base:
    def __init__(self,pruning_ratio,pruning_method):
        """
        Initialize the pruning engine base class

        Args:
        - pruning_ratio: The pruning ratio to be applied (e.g. pruning_ratio = 0.2 retained)
        - pruning_method: The chosen pruning method

        Return: None
        """
        self.pruning_ratio = 1-pruning_ratio        # 1 - pruning_ratio to obtain percentage of parameters to prune
        self.pruning_method = pruning_method        # pruning method to be applied
        self.mask_number = 1e10                     # mask_number is a placeholder set to a large number to easily identify elements that need to be pruned
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def base_remove_filter_by_index(self,weight,remove_filter_idx,bias=None,mean=None,var=None,linear=False):       
        """
        Remove the specified filters from the layer's weight, bias, mean, and var tensors

        Args:
        - weight: The weight tensor of the layer
        - remove_filter_idx: List of indices of filters to be removed
        - bias: The bias tensor of the layer
        - mean: The mean tensor of the Batch Norm layer
        - var: The variance tensor of the Batch Norm layer
        - linear: A boolean flag indicating whether the layer is a Linear layer

        Return:
        - weight: The updated weight tensor after removing the filters
        - bias: The updated bias tensor after removing the filters
        - mean: The updated mean tensor after removing the filters
        - var: The updated variance tensor after removing the filters
        """
        if mean is not None:        # if mean tensor exists -> batch norm layer
            mask_tensor = torch.tensor(self.mask_number,device=self.device)     # create a mask tensor to mark parameters to be pruned, contains a single element mask_number

            for idx in remove_filter_idx:           # iterate over each element of remove_filter_idx list
                # mark the weight, bias, mean, and variance tensors at the corresponding index to mask_tensor for pruning
                # the filter at index idx will be marked for pruning
                weight[idx.item()] = mask_tensor
                bias[idx.item()] = mask_tensor
                mean[idx.item()] = mask_tensor 
                var[idx.item()] = mask_tensor
            
            # removal of masked filters by excluding elements marked by mask_tensor
            weight = weight[weight != mask_tensor]      # weight tensor = weight tensor w/o masked filters
            bias = bias[bias != mask_tensor]
            mean = mean[mean != mask_tensor]
            var = var[var != mask_tensor]

            return weight,bias,mean,var
        elif bias is not None:      # if bias tensor exists -> linear or conv layer
            mask_tensor = torch.tensor(self.mask_number,device=self.device)         # create a mask tensor to mark parameters to be pruned, contains a single element mask_number
            mask_tensor = mask_tensor.repeat(list(weight[0].size()))                # adjust mask_tensor dimensions to match size of weight tensor, repeat() is used to extend mask_number value across first filter size weight[0].size()
            bias_mask_tensor = torch.tensor(self.mask_number,device=self.device)    # create a bias mask tensor used for bias values
            
            for idx in remove_filter_idx:           # iterate over each element of remove_filter_idx list
                # mark the weight and bias tensors at the corresponding index to mask_tensor and bias_mask_tensor, respectively
                weight[idx.item()] = mask_tensor
                bias[idx.item()] = bias_mask_tensor

            if linear is False:     # if not a linear layer -> conv layer (multiple dimensions)
                # create a boolean tensor nonMaskRows_weight that calculates the absolute difference between the summed weights in a weight tensor and the summed weights of mask_tensor
                # a fully masked filter has all its weights set to mask_number; weight tensor sum = mask_tensor sum and the difference is 0 which returns False
                # unmasked filters return True and masked filters return False
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(1,2,3)) - torch.abs(mask_tensor).sum(dim=(0,1,2))) > self.mask_number
            else:                   # linear layer (single dimension)
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=1) - torch.abs(mask_tensor).sum(dim=0)) > self.mask_number
            
            # filter the tensors by excluding elements marked by mask_tensor
            weight = weight[nonMaskRows_weight]     # weight tensor = weight tensor w/o masked filters for nonMaskRows_weight = True
            bias = bias[bias != self.mask_number]   # equivalent to bias = bias[bias != bias_mask_tensor]
            
            return weight,bias
        else:   # bias, mean, and var tensors don't exist
            mask_tensor = torch.tensor(self.mask_number,device=self.device)     # create a mask tensor
            mask_tensor = mask_tensor.repeat(list(weight[0].size()))            # adjust size of mask tensor to match filter size (weight[0].size() gives first filter size)

            for idx in remove_filter_idx:           # iterate over remove_filter_idx list
                weight[idx.item()] = mask_tensor    # set weight tensor at idx to mask_tensor

            if linear is False:     # if conv layer
                # create boolean tensor
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(1,2,3)) - torch.abs(mask_tensor).sum(dim=(0,1,2))) > self.mask_number
            else:
                nonMaskRows_weight = abs(torch.abs(weight).sum(dim=1) - torch.abs(mask_tensor).sum(dim=1)) > self.mask_number

            weight = weight[nonMaskRows_weight]     # weight tensor = weight tensor w/o masked filters for nonMaskRows_weight = True

            return weight

    def base_remove_kernel_by_index(self,weight,remove_filter_idx,linear=False):
        """
        Remove the specified kernels from the layer's weight tensor

        Args:
        - weight: The weight tensor of the layer
        - remove_filter_idx: List of indices of kernels to be removed
        - linear: A boolean flag indicating whether the layer is a Linear layer

        Return:
        - weight: The updated weight tensor after removing the kernels
        """
        mask_tensor = torch.tensor(self.mask_number,device=self.device)     # create a mask tensor
        mask_tensor = mask_tensor.repeat(list(weight[0][0].size()))         # adjust size of mask tensor to match kernel size (weight[0][0].size() gives first kernel size)
        
        for idx in remove_filter_idx:           # iterate over remove_filter_idx list
            weight[:,idx.item()] = mask_tensor  # set the entire column at idx of the weight tensor to mask_tensor, the kernel at idx across all filters are marked for pruning

        if (len(remove_filter_idx) != 0 and linear == False):   # if remove_filter_idx list exists and it's a conv layer
            # create a boolean tensor nonMaskRows_weight to calculate absolute difference between weight sum and mask_tensor sum
            # True when non fully masked and False when masked
            nonMaskRows_weight = abs(torch.abs(weight).sum(dim=(2,3)) - torch.abs(mask_tensor).sum(dim=(0,1))) > 0.0001 
            weight = weight[:,nonMaskRows_weight[0]]    # weight tensor = weight tensor w/o masked kernels (removal of masked kernels)

        if (linear != False):   # if linear layer
            # weight tensor is 2D, rows = output neurons, columns = input neurons
            # remove marked columns of the weight tensor and keep those not marked for pruning
            weight = weight[:,weight[1]!=mask_tensor]   # weight tensor = weight tensor w/o columns (input neurons) marked by mask_tensor

        return weight