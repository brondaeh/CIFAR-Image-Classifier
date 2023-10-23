'''ResNet in PyTorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    '''
    BasicBlock (Residual Block) is a building block class of ResNet
    2 conv layers and an optional skip connection per block
    '''
    expansion = 1   # expansion adjusts the number of output channels in conv layers

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        # first conv layer 3x3 filter
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # second conv layer 3x3 filter
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # skip connection if stride != 1 (downsampling, feature map dimensions are reduced)
        self.shortcut = nn.Sequential() 
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), # 1x1 conv layer
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # conv1 -> batchnorm -> ReLU
        out = self.bn2(self.conv2(out))         # conv2 -> batchnorm
        out += self.shortcut(x)                 # adds skip connection
        out = F.relu(out)                       # ReLU activation
        return out


class Bottleneck(nn.Module):
    '''
    Bottleneck is an alternative residual block class for ResNet models
    Helps to reduce computational complexity of deeper ResNet models
    3 conv layers: dimension reduction, feature transformation, dimension restoration + skip connection
    '''
    expansion = 4   # expansion adjusts the number of output channels in conv layers

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # first conv layer: 1x1 filter to reduce the number of channels of input feature map to reduce complexity
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # second conv layer: 3x3 filter to capture features, patterns, etc.
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # third conv layer: 1x1 filter to increase the number of channels of the input feature map (dimension restoration)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # skip connection if stride != 1 (downsampling, feature map dimensions are reduced)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False), # 1x1 conv layer
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # conv1 -> batchnorm -> ReLU
        out = F.relu(self.bn2(self.conv2(out))) # conv2 -> batchnorm -> ReLU
        out = self.bn3(self.conv3(out))         # conv3 -> batchnorm
        out += self.shortcut(x)                 # adds skip connection
        out = F.relu(out)                       # ReLU activation
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64     # in_planes tracks the number of input channels in each layer

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)   # first conv layer 3 input channels, 64 output channels, 3x3 filter
        self.bn1 = nn.BatchNorm2d(64)

        # the subsequent layers are created with _make_layer()
        # parameters: block class, output planes, number of blocks, stride
        # the resulting blocks are stored in a nn.Sequential container
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)     # list of strides for each block: first element is the provided stride parameter, all others are 1
        layers = []                                 # layers list to store 
        for stride in strides:                      # iterates thru each element in strides
            layers.append(block(self.in_planes, planes, stride))    # a block instance is created for each iteration and appended to the layers list
            self.in_planes = planes * block.expansion               # input channels of the next block is matched with the current number of output channels
        # returns a nn.Sequential container of the constructed blocks containing conv layers
        # *layers unpacks the list and passes each individual layer to nn.Sequential which will arrange them in order
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   # conv1 -> batchnorm -> ReLU
        out = self.layer1(out)                  # first sequential layer of blocks
        out = self.layer2(out)                  # second sequential layer of blocks
        out = self.layer3(out)                  # third sequential layer of blocks
        out = self.layer4(out)                  # fourth sequential layer of blocks
        out = F.avg_pool2d(out, 4)              # average pooling layer to reduce feature map dimensions to 1
        out = out.view(out.size(0), -1)         # flattens feature map to 1D vector
        out = self.linear(out)                  # FC layer connects all inputs to class scores
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
