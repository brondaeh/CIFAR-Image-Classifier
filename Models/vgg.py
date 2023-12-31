'''VGG (Visual Geometry Group) Models in PyTorch'''

import torch
import torch.nn as nn

# Configuration dictionary for each VGG model variant with layer specifications (channel sizes, M for max-pooling)
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):           # vgg_name specifies variant
        super(VGG, self).__init__()
        self.version = vgg_name
        self.features = self._make_layers(cfg[vgg_name])    # call _make_layers() to create conv layers based on vgg_name from cfg
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.features(x)                              # x passed thru conv layers
        out = out.view(out.size(0), -1)                     # x is flattened
        out = self.classifier(out)                          # x is passed thru final FC layer
        return out

    def _make_layers(self, cfg):                            # method to construct conv layers based on chosen cfg
        layers = []                                         # layers is a list that stores each conv layer
        in_channels = 3                                     # 3 input channels for CIFAR-10 dataset
        for x in cfg:                                       # iterate over each element in cfg
            if x == 'M':                                    # if element is 'M', append a max pool layer to the layers list
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:                                           # element is an integer, append a conv layer, batchnorm, and ReLU to the layers list
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x                             # set input channels of next layer to current number of output channels
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]   # append an average pool layer to the layers list
        return nn.Sequential(*layers)                       # return a nn.Sequential container of all conv layers
