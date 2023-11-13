# pytorch-cifar10

## Introduction
In this project, multiple CNN architectures are used to train image classification models on the CIFAR-10 dataset. Model compression methods like network pruning are also explored to reduce the size of the neural network by removing redundant parameters which leads to improved computational efficiency during inference.

## Accuracy

|   Model   | Accuracy |
|-----------|----------|
|[VGG16](Model_Details/vgg16_details.md)      |  92.55%  |
|[MobileNetV1](Model_Details/mobilenetv1_details.md)|  91.28%  |
|[MobileNetV2](Model_Details/mobilenetv2_details.md)|  93.39%  |
|[ResNet18](Model_Details/resnet18_details.md)|  93.96%  |
|VGG16 (Pruned)|          |

## References
- [CNN-Pruning-Engine](https://github.com/MIC-Laboratory/CNN-Pruning-Engine)
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
