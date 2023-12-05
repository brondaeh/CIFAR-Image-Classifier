# VGG16 Model Details
## Training parameters to achieve 92.55% accuracy on CIFAR-10 dataset

- **Batch Size:** 256
- **Number of Epochs:** 60
- **Data Augmentations:**
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
- **Learning Rate:** Starts at lr=0.1, changes based on scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60, eta_min=0.001)
- **L2 Regularization:** optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)

## Fine-tuning parameters after L1 Norm uniform pruning at 10%

- **Batch Size:** 256
- **Number of Epochs:** 30
- **Data Augmentations:**
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
- **Learning Rate:** Starts at lr=0.05, changes based on scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=30, eta_min=0.0005)
- **L2 Regularization:** optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)