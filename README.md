# CIFAR Image Classifier

## Introduction
In this project, multiple CNN architectures are used to train image classification models on the CIFAR dataset. Model compression methods like network pruning are also explored to reduce the size of the neural network by removing redundant parameters which leads to improved computational efficiency during inference.

## Accuracy
|  Dataset  |   Model   | Pruning Method | Pruning Ratio (%) | Params (M) | Top1 Acc (%) |
|-----------|-----------|----------------|-------------------|------------|--------------|
|  CIFAR10  |[VGG16](Model_Details/vgg16_details.md)             |  -                |  -  | 14.73 |  94.45  |
|  CIFAR10  |[MobileNetV1](Model_Details/mobilenetv1_details.md) |  -                |  -  | 3.22  |  91.28  |*
|  CIFAR10  |[MobileNetV2](Model_Details/mobilenetv2_details.md) |  -                |  -  | 2.3   |  93.39  |*
|  CIFAR10  |[ResNet18](Model_Details/resnet18_details.md)       |  -                |  -  | 11.17 |  93.96  |*
|  CIFAR10  |[VGG16](Model_Details/vgg16_details.md)             |  L1 Norm Uniform  | 10  | 11.89 |  92.19  |
|  CIFAR100 |[VGG16](Model_Details/vgg16_details.md)             |  -                |  -  |       |         |
|  CIFAR100 |[VGG16](Model_Details/vgg16_details.md)             |  L1 Norm Uniform  | 10  |       |         |

## References
- [CNN-Pruning-Engine](https://github.com/MIC-Laboratory/CNN-Pruning-Engine)
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
