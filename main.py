'''Training neural network models on CIFAR-10 dataset with PyTorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import time
import matplotlib.pyplot as plt

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data Transformation
print("--> Preparing Data...")

transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
)

transform_test = transform_train

# Loading CIFAR-10 Dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Data Loaders
batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Definition of Classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Method for Visualizing Training Images
def imshow(img):                                    # takes input parameter pytorch tensor of 1 image
    img = img / 2 + 0.5                             # unnormalize
    npimg = img.numpy()                             # convert to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))      # change order of dimensions from pytorch (channels, height, wdith) -> plt (height, width, channels)
    plt.show()

# Model Initialization
print("--> Initializing Model...")

net = VGG('VGG11')
net = net.to(device)                    # moves model to device to ensure the next computations are performed on the specified device
if device == 'cuda':
    net = torch.nn.DataParallel(net)    # wraps model with DataParallel to parallelize training process on GPUs if available
    cudnn.benchmark = True              # enables cuDNN benchmarking mode for best algorithm during convolutional operations

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()                                   # applies softmax   
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)     # used to update parameters (weights and biases) to minimize loss function

# Training the model
print("--> Training Model...")

num_epochs = 2                  # set number of epochs (number of times dataset is seen by model)
tick = time.time()              # record start time to track total training time

for epoch in range(num_epochs): # training loop processes entire training dataset num_epochs times with the model
    running_loss = 0.0          # tracks running loss to be printed

    for i, data in enumerate(trainloader, 0):   # iterate thru mini-batches, i = current mini-batch, data = trainloader data
        inputs, labels = data                   # unpack mini-batch data to input image and corresponding ground-truth labels [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()                   # zero all parameter gradients
        outputs = net(inputs)                   # forward pass to get predictions from model
        loss = criterion(outputs, labels)       # loss calculation compares predicted outputs and ground-truth labels
        loss.backward()                         # backward pass to compute gradients
        optimizer.step()                        # update model's weights and biases with gradients
        
        running_loss += loss.item()             # update current mini-batch loss to running loss
        if i % 2000 == 1999:                    # print avg loss over the last 2000 mini-batches for every 2000
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0                  # reset running loss

tock = time.time()
trainingTime = tock - tick
print(f"Training completed in {trainingTime:.2f} seconds.")

# Test the Model on the Entire Test Dataset
print("--> Testing Model...")

correct = 0     # tracks correct predictions
total = 0       # tracks total images covered

with torch.no_grad():               # disable gradient calculations since we're not training
    for data in testloader:         # iterate over mini-batches in test dataset
        inputs, labels = data       # assign input image and ground truth label for each mini-batch
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)       # forward pass: test images sent through network for predictions
        _, predicted = torch.max(outputs.data, 1)       # predicted class indices are obtained

        # updates total image count for each mini-batch
        # labels is the ground truth class labels 
        total += labels.size(0)
        
        # compares predicted class indices with ground truth class indicies element-wise (T/F 1/0), returns a Boolean tensor
        # .sum() calculates the number of correct predictions since 1 is true and 0 is false
        # .item() extracts numerical value of the summation (integer)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on 10000 test images: {100 * correct // total}%')

# Check Accuracy for Each Class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')
