# CIFAR Image Classifier

## Introduction
In this project, multiple CNN architectures are used to train image classification models on the CIFAR dataset. Model compression methods like network pruning are also explored to reduce the size of the neural network by removing redundant parameters which leads to improved computational efficiency during inference.

## Performance
|  Dataset  |   Model   | Pruning Method | Pruning Ratio (%) | Params (M) | Top1 Acc (%) |
|-----------|-----------|----------------|-------------------|------------|--------------|
|  CIFAR10  |[VGG16](Model_Details/vgg16_details.md)             |  -                |  -  | 33.65 | 94.71 |
|  CIFAR10  |[VGG16](Model_Details/vgg16_details.md)             |  L1 Norm Uniform  | 10  | 28.74 | 92.19 |
|  CIFAR10  |[MobileNetV1](Model_Details/mobilenetv1_details.md) |  -                |  -  | 3.22  | 91.53 |
|  CIFAR10  |[MobileNetV2](Model_Details/mobilenetv2_details.md) |  -                |  -  | 2.3   | 94.44 |
|  CIFAR10  |[ResNet18](Model_Details/resnet18_details.md)       |  -                |  -  | 11.17 | 95.42 |
|  CIFAR100 |[VGG16](Model_Details/vgg16_details.md)             |  -                |  -  | 34.02 | 73.95 |
|  CIFAR100 |[VGG16](Model_Details/vgg16_details.md)             |  L1 Norm Uniform  | 10  | 29.07 | 63.92 |
|  CIFAR100 |[MobileNetV1](Model_Details/mobilenetv1_details.md) |  -                |  -  | 3.31  | 68.18 |
|  CIFAR100 |[MobileNetV2](Model_Details/mobilenetv2_details.md) |  -                |  -  | 2.41  | 75.96 |
|  CIFAR100 |[ResNet18](Model_Details/resnet18_details.md)       |  -                |  -  | 11.22 | 77.32 |

## Getting Started
1. Clone the repository to the desired directory:
```bash
git clone https://github.com/brondaeh/CIFAR-Image-Classifier.git
```
2. Install the required libraries to your environment:
```bash
pip install -r requirements.txt
```
3. Modify the configuration parameters in config.yaml.
    - NOTE:
    - Set enable_gpu to True if you have a cuda enabled GPU for training.
    - By default L1 norm pruning is only applicable for VGG16, keep pruning_flag to False for other models.
4. Start training and testing by running main.py.

## File Structure
```bash
CIFAR-Image-Classifier
├── Model_Details
│   ├── mobilenetv1_details.md
│   ├── mobilenetv2_details.md
│   ├── resnet18_details.md
│   └── vgg16_details.md
├── Models
│   ├── mobilenetv1.py
│   ├── mobilenetv2.py
│   ├── resnet.py
│   └── vgg.py
├── Pruner
│   ├── Pruning_Criterion
│   |   ├── KMean
│   |   |   └── KMean_base.py
│   |   ├── L1norm
│   |   |   └── L1norm.py
│   |   ├── Taylor
│   |   |   └── Taylor.py
│   ├── pruning_engine_base.py
│   └── pruning_engine.py
├── Pruning_Functions
│   └── prune_vgg.py
├── config.yaml
├── main.py
├── README.md
├── requirements.txt
├── utils.py
```

## References
- [CNN-Pruning-Engine](https://github.com/MIC-Laboratory/CNN-Pruning-Engine)
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
