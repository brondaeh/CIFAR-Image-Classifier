'''MobileNetV1 Model in PyTorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''
    Block is a building block class of MobileNet
    Depthwise conv + Pointwise conv
    '''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()

        # depthwise convolution
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)

        # pointwise convolution
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))       # depthwise -> batchnorm -> ReLU
        out = F.relu(self.bn2(self.conv2(out)))     # pointwise -> batchnorm -> ReLU
        return out

class MobileNet(nn.Module):
    # configuration dictionary to describe each block
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)   # first standard conv layer, 3 channels, 32 outputs
        self.bn1 = nn.BatchNorm2d(32)                                                   # batchnorm applied to each output
        self.layers = self._make_layers(in_planes=32)                                   # call _make_layers() to construct sequence of blocks (depthwise seperable conv)
        self.linear = nn.Linear(1024, num_classes)                                      # FC layer connects 1024 inputs to 10 outputs (10 classes)

    def _make_layers(self, in_planes):                              # in_planes = number of inputs for first block (32)
        layers = []                                                 # layers is a list that stores the layers of each conv block
        for x in self.cfg:                                          # iterates over each element in cfg
            out_planes = x if isinstance(x, int) else x[0]          # if integer, number of outputs = x; if tuple, number of outputs = first value x[0]
            stride = 1 if isinstance(x, int) else x[1]              # if integer, stride = 1; if tuple, stride = second value x[1]
            layers.append(Block(in_planes, out_planes, stride))     # creates a new Block instance with specified inputs, outputs, and stride; append to layers list
            in_planes = out_planes                                  # match input planes of the next block to current output planes
        return nn.Sequential(*layers)                               # returns an nn.Sequential container with all blocks stored in layers

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # first full conv layer -> batchnorm -> ReLU
        out = self.layers(out)                  # next depthwise seperable conv layers defined by Block
        out = F.avg_pool2d(out, 2)              # average pool to reduce spatial dimensions
        out = out.view(out.size(0), -1)         # flatten feature map to 1D tensor
        out = self.linear(out)                  # pass flattened tensor thru FC layer to produce class scores
        return out