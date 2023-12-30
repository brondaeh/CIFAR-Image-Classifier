# MobileNetV2 Model Details

## CIFAR-10 Training Parameters (Top1 94.44%)

- **Number of Epochs:** 200
- **Batch Size:** 256
- **Learning Rate:** Starts at 0.01 and changes based on Cosine Annealing scheduler
- **Momentum:** 0.9
- **L2 Regularization Weight Decay:** 0.001
- **Data Augmentations:**
    ```python
    transform_train = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.autoaugment.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=dataset_mean, std=dataset_std),
    ])
    ```
    ```python
    transform_test = transforms.Compose([
        transforms.RandomCrop(input_size,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])
    ```

## CIFAR-100 Training Parameters (Top1 75.96%)

- **Number of Epochs:** 200
- **Batch Size:** 256
- **Learning Rate:** Starts at 0.01 and changes based on Cosine Annealing scheduler
- **Momentum:** 0.9
- **L2 Regularization Weight Decay:** 0.001
- **Data Augmentations:**
    ```python
    transform_train = transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.autoaugment.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.Normalize(mean=dataset_mean, std=dataset_std),
    ])
    ```
    ```python
    transform_test = transforms.Compose([
        transforms.RandomCrop(input_size,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])
    ```
    