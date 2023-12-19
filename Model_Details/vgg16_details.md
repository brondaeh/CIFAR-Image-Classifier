# VGG16 Model Details

## Training parameters to achieve 94.45% top1 accuracy on CIFAR-10 dataset

- **Number of Epochs:** 200
- **Batch Size:** 256
- **Data Augmentations:**
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
    transform_test = transforms.Compose([
        transforms.RandomCrop(input_size,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])
- **Learning Rate:** Starts at lr=0.01 and changes based on Cosine Annealing scheduler
- **L2 Regularization and Momentum:** optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

## Fine-tuning parameters to achieve 92.19% top1 accuracy after L1 Norm uniform pruning at 10%

- **Number of Epochs:** 40
- **Batch Size:** 256
- **Data Augmentations:**
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
    transform_test = transforms.Compose([
        transforms.RandomCrop(input_size,padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_mean,std=dataset_std)
    ])
- **Learning Rate:** Starts at lr=0.01 and changes based on Cosine Annealing scheduler
- **L2 Regularization and Momentum:** optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)