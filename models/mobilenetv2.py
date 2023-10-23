'''MobileNetV2 Model in PyTorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''
    Block (Inverted Residual Block) is a building block class of MobileNetV2
    3 conv layers: expand + depthwise + pointwise
    '''
    def __init__(self, in_planes, out_planes, expansion, stride):   # a Block instance takes 4 parameters from the _make_layers() method
        super(Block, self).__init__()
        self.stride = stride                                        

        planes = expansion * in_planes                              # changes the number of intermediate channels in the block with expansion factor

        # expand: pointwise conv layer to expand number of channels of the input feature map, in_planes < out_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # depthwise: dw conv layer to filter data from input feature map, in_planes = out_planes
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # pointwise: pw conv layer to reduce channel size in feature map, in_planes > out_planes
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        # skip connection to bypass conv layers if stride is 1 (no downsampling) and in_planes != out_planes
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),   # pw conv layer to ensure in_planes = out_planes since there is no downsampling
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))                   # pw conv layer -> batchnorm -> ReLU
        out = F.relu(self.bn2(self.conv2(out)))                 # dw conv layer -> batchnorm -> ReLU
        out = self.bn3(self.conv3(out))                         # pw conv layer -> batchnorm
        out = out + self.shortcut(x) if self.stride==1 else out # add skip connection
        return out


class MobileNetV2(nn.Module):
    # configuration dictionary for layers in Block (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)       # first full conv layer 3 in channels, 32 out channels, 3x3 filter
        self.bn1 = nn.BatchNorm2d(32)                                                       # batchnorm for first conv layer
        self.layers = self._make_layers(in_planes=32)                                       # calls _make_layers() to construct Blocks with first input size = 32
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)   # pointwise conv layer 320 in channels, 1280 out channels, 1x1 filter
        self.bn2 = nn.BatchNorm2d(1280)                                                     # batchnorm for pointwise conv layer
        self.linear = nn.Linear(1280, num_classes)                                          # FC layer connecting 1280 inputs to 10 outputs (10 class scores)

    def _make_layers(self, in_planes):
        layers = []                                                             # layers is a list that stores the Block instances of conv layers
        for expansion, out_planes, num_blocks, stride in self.cfg:              # iterates thru tuple elements in cfg
            strides = [stride] + [1]*(num_blocks-1)                             # strides is a list, first element is set to the stride from cfg and the rest are set to 1
            for stride in strides:                                              # iterates thru the strides list
                layers.append(Block(in_planes, out_planes, expansion, stride))  # Block instances are created to define subsequent conv layers and appended to layers list
                in_planes = out_planes                                          # match the input planes of the next Block to the current number of output planes
        return nn.Sequential(*layers)                                           # return a nn.Sequential container that stores all the Blocks in the layers list

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))       # first full conv layer -> batchnorm -> ReLU
        out = self.layers(out)                      # next conv layers defined by Block
        out = F.relu(self.bn2(self.conv2(out)))     # pointwise conv layer -> batchnorm -> ReLU
        out = F.avg_pool2d(out, 4)                  # avg pool layer to downsample NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = out.view(out.size(0), -1)             # flatten input tensor
        out = self.linear(out)                      # FC layer connects to 10 class scores
        return out
