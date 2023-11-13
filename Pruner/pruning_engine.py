'''Pruning Engine Class in PyTorch'''

import torch

from copy import deepcopy
from .pruning_engine_base import pruning_engine_base
from .Pruning_Criterion.L1norm.L1norm import L1norm
from .Pruning_Criterion.Taylor.Taylor import Taylor
# from .pruning_criterion.KMean.K_L1norm import K_L1norm
# from .pruning_criterion.KMean.K_Taylor import K_Taylor
# from .pruning_criterion.KMean.K_Distance import K_Distance

class pruning_engine(pruning_engine_base):
    def __init__(self,pruning_method,pruning_ratio = 0,individual = False,**kwargs):
        """
        Initialize the pruning engine class which inherits from pruning engine base

        Args:
        - pruning_method: The pruning method to be used
        - pruning_ratio: The pruning ratio to be applied (percentage of filters pruned)

        Return: None

        Logic:
        - Initialize the pruning engine with the specified pruning method and pruning ratio
        """
        super().__init__(pruning_ratio,pruning_method)
        
        # choose the pruning criterion based on specified pruning method
        if (self.pruning_method == "L1norm"):   # if L1norm
            self.l1norm_pruning = L1norm()                                  # create an instance of L1norm and assign to l1norm_pruning
            self.pruning_criterion = self.l1norm_pruning.L1norm_pruning     # assign pruning criterion method of L1norm to pruning_criterion
        elif (self.pruning_method == "Taylor"): # if Taylor
            self.taylor_pruning = Taylor(                                   # create an instance of Taylor with the following args and assign to taylor_pruning
                tool_net=kwargs["tool_net"],
                total_layer=kwargs["total_layer"], 
                taylor_loader=kwargs["taylor_loader"],
                total_sample_size=kwargs["total_sample_size"], 
                hook_function=kwargs["hook_function"])
            self.taylor_pruning.clear_mean_gradient_feature_map()
            self.taylor_pruning.Taylor_add_gradient()
            self.taylor_pruning.store_grad_layer(kwargs["layer_store_private_variable"])
            self.pruning_criterion = self.taylor_pruning.Taylor_pruning     # assign pruning criterion method of Taylor to pruning_criterion

        # NOTE KMean pruning methods exlcuded
        # elif (self.pruning_method == "K-L1norm"):
        #     self.K_L1norm_pruning = K_L1norm(list_k=kwargs["list_k"],pruning_ratio=self.pruning_ratio)
        #     self.K_L1norm_pruning.store_k_in_layer(kwargs["layer_store_private_variable"])
        #     self.pruning_criterion = self.K_L1norm_pruning.Kmean_L1norm
        # elif (self.pruning_method == "K-Taylor"):
        #     self.K_Taylor_pruning = K_Taylor(
        #         list_k=kwargs["list_k"],
        #         pruning_ratio=self.pruning_ratio,
        #         tool_net=kwargs["tool_net"],
        #         total_layer=kwargs["total_layer"], 
        #         taylor_loader=kwargs["taylor_loader"],
        #         total_sample_size=kwargs["total_sample_size"], 
        #         hook_function=kwargs["hook_function"],
        #         layer_store_grad_featuremap=kwargs["layer_store_private_variable"])
        #     self.K_Taylor_pruning.store_k_in_layer(kwargs["layer_store_private_variable"])
        #     self.pruning_criterion = self.K_Taylor_pruning.Kmean_Taylor
        # elif (self.pruning_method == "K-Distance"):
        #     self.K_Distance_Pruning = K_Distance(list_k=kwargs["list_k"],pruning_ratio=self.pruning_ratio)
        #     self.K_Distance_Pruning.store_k_in_layer(kwargs["layer_store_private_variable"])
        #     self.pruning_criterion = self.K_Distance_Pruning.Kmean_Distance

        # remove_filter_idx_history is a dictionary that tracks removed filters across layers
        self.remove_filter_idx_history = {
            "previous_layer":None,
            "current_layer":None
        }

        # individual determines whether pruning is performed independently (True) for each layer or collectively (False)
        self.individual = individual


    def set_layer(self,layer,main_layer=False):
        """
        Set the current layer for pruning

        Args:
        - layer: The layer to be pruned

        Return: None

        Logic:
        - Set the current layer to the given layer for further pruning operations
        """
        # assigns a deep copy of the input layer to copy_layer
        self.copy_layer = deepcopy(layer)
        
        if main_layer:  # if the current layer is main_layer
            if self.individual:     # if individual pruning is chosen
                # reset remove_filter_idx_history
                self.remove_filter_idx_history = {
                    "previous_layer":None,
                    "current_layer":None
                }
            
            # update remove_filter_idx_history
            self.remove_filter_idx_history["previous_layer"] = self.remove_filter_idx_history["current_layer"]  # set previous layer to current layer (indices potentially pruned from previous layer)
            self.remove_filter_idx_history["current_layer"] = None                                              # reset current layer indices for the upcoming layer

            # pruning_criterion is applied to copy_layer to compute the indices of filters for potential pruning, stored to remove_filter_idx list
            remove_filter_idx = self.pruning_criterion(self.copy_layer)
            # calculate the number of filters to prune and store to number_pruning_filter
            # len(remove_filter_idx) is the total number of potentially pruned filters and self.pruning_ratio is the percentage of filters to keep
            number_pruning_filter = int(len(remove_filter_idx) * self.pruning_ratio)
            # self.remove_filter_idx is updated to store only the section of filter indices of remove_filter_idx from index = number_pruning_filter to the end
            # all filters before number_pruning_filter are excluded from pruning and this list holds the indices we want to prune
            self.remove_filter_idx = remove_filter_idx[number_pruning_filter:]

            if (self.remove_filter_idx_history["previous_layer"] is None):  # if layer is the first layer being pruned (no pruning history)
                self.remove_filter_idx_history["previous_layer"] = self.remove_filter_idx   # update previous layer of remove_filter_idx_history with filter indices of the current layer

            self.remove_filter_idx_history["current_layer"] = self.remove_filter_idx    # update current layer with filter indices of the current layer

        return True
    
    
    def set_pruning_ratio(self,pruning_ratio):
        """
        Set the pruning ratio for the current layer

        Args:
        - pruning_ratio: The pruning ratio to be applied to the current layer

        Return: None

        Logic:
        - Set the pruning ratio for the current layer to the specified value
        """
        self.pruning_ratio = 1-pruning_ratio

        # NOTE KMean prunig methods excluded
        # if "K_L1norm_pruning" in self.__dict__:
        #     self.K_L1norm_pruning.set_pruning_ratio(1-pruning_ratio)
        # if "K_Distance_Pruning" in self.__dict__:
        #     self.K_Distance_Pruning.set_pruning_ratio(1-pruning_ratio)
        # if "K_Taylor_pruning" in self.__dict__:
        #     self.K_Taylor_pruning.set_pruning_ratio(1-pruning_ratio)


    def get_remove_filter_idx(self):
        """
        Get the indices of removed filters

        Args: None

        Return:
        - remove_filter_idx: The indices of filters removed during pruning

        Logic:
        - Get the indices of filters that have been removed during the pruning process
        """
        return self.remove_filter_idx_history


    def remove_conv_filter_kernel(self):
        """
        Remove filters and corresponding kernels from the convolutional layer

        Args: None

        Return: None

        Logic:
        - Remove filters and corresponding kernels from the current convolutional layer based on the pruning ratio
        """
        if self.copy_layer.bias is not None:    # if there is a bias tensor in the conv layer -> not a batchnorm layer
            # remove filters by index and update weight and bias
            self.copy_layer.weight.data,self.copy_layer.bias.data = self.base_remove_filter_by_index(weight=self.copy_layer.weight.data.clone(),remove_filter_idx=self.remove_filter_idx_history["current_layer"],bias=self.copy_layer.bias.data.clone())
            # remove kernels by index and update weight
            self.copy_layer.weight.data = self.base_remove_kernel_by_index(weight=self.copy_layer.weight.data.clone(), remove_filter_idx=self.remove_filter_idx_history["previous_layer"])
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]
        else:   # batchnorm layer
            self.copy_layer.weight.data = self.base_remove_filter_by_index(weight=self.copy_layer.weight.data.clone(),remove_filter_idx=self.remove_filter_idx_history["current_layer"])
            self.copy_layer.weight.data = self.base_remove_kernel_by_index(weight=self.copy_layer.weight.data.clone(), remove_filter_idx=self.remove_filter_idx_history["previous_layer"])
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]
        
        return self.copy_layer
    
    def remove_Bn(self,remove_filter_idx):
        """
        Remove filters from the Batch Normalization layer.

        Args: None

        Return: None

        Logic:
        - Remove filters from the Batch Normalization layer based on the pruning ratio.
        """        
        self.copy_layer.weight.data,\
        self.copy_layer.bias.data,\
        self.copy_layer.running_mean.data,\
        self.copy_layer.running_var.data = self.base_remove_filter_by_index(
            self.copy_layer.weight.data.clone(), 
            remove_filter_idx,
            bias=self.copy_layer.bias.data.clone(),
            mean=self.copy_layer.running_mean.data.clone(),
            var=self.copy_layer.running_var.data.clone()
            )
        self.copy_layer.num_features = self.copy_layer.weight.shape[0]

        return self.copy_layer
    
    def remove_filter_by_index(self,remove_filter_idx,linear=False,group=False):
        """
        Remove filters from the current layer based on the given indices.

        Args:
        - idx: Indices of filters to be removed.

        Return: None

        Logic:
        - Remove filters from the current layer based on the given indices.
        """
        if self.copy_layer.bias is not None:
            self.copy_layer.weight.data,\
            self.copy_layer.bias.data = self.base_remove_filter_by_index(
                weight=self.copy_layer.weight.data.clone(),
                remove_filter_idx=remove_filter_idx,
                bias=self.copy_layer.bias.data.clone(),
                linear=linear
                )
        else:
            self.copy_layer.weight.data = self.base_remove_filter_by_index(
                weight=self.copy_layer.weight.data.clone(),
                remove_filter_idx=remove_filter_idx,
                linear=linear
                )
            
        if linear:
            self.copy_layer.out_features = self.copy_layer.weight.shape[0]
        else:
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]
        
        if group:
            self.copy_layer.groups = self.copy_layer.weight.shape[0]
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]
            self.copy_layer.out_channels = self.copy_layer.weight.shape[0]

        return self.copy_layer

    def remove_kernel_by_index(self,remove_filter_idx,linear=False):
        """
        Remove kernels from the current layer based on the given indices.

        Args:
        - idx: Indices of kernels to be removed.

        Return: None

        Logic:
        - Remove kernels from the current layer based on the given indices.
        """
        self.copy_layer.weight.data = self.base_remove_kernel_by_index(
            weight=self.copy_layer.weight.data.clone(),
            remove_filter_idx=remove_filter_idx,
            linear=linear
            )
        
        if linear:
            self.copy_layer.in_features = self.copy_layer.weight.shape[1]
        else:
            self.copy_layer.in_channels = self.copy_layer.weight.shape[1]

        return self.copy_layer