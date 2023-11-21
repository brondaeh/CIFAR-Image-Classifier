'''Training and pruning neural network models on CIFAR-10 dataset in PyTorch'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import time

from Models import *
from Pruner import *
from utils import save_learning_curve, model_complexity, save_model, save_pruned_accuracy_curve
from prune_vgg import uniform_prune_vgg16


total_train_loss = []       # list to track total training loss for each epoch; will be used for plotting a learning curve
total_test_loss = []        # list to track total test loss
total_test_accuracy = []    # list to track total test accuracy 

def train(model, criterion, optimizer, batch_size):
    '''
    Trains the model with forward and backprop

    Args:
    - model: the desired model for training
    - criterion: loss function (usually CrossEntropyLoss)
    - optimizer: algorithm used to minimize loss function (usually SGD)
    - batch_size: the number of training examples used in one training iteration

    Return:
    - avg_train_loss: average training loss for the current epoch
    '''
    train_loss = 0.0                                            # tracks the running training loss for each epoch
    model.train()                                               # enable training mode on the model

    for batch_index, data in enumerate(trainloader, 0):         # iterate thru mini-batches, batch_index = current mini-batch, data = trainloader data
        inputs, labels = data                                   # unpack mini-batch data to input image and corresponding ground-truth labels [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)   # send inputs and labels to specified device for computations

        optimizer.zero_grad()                                   # zero all parameter gradients
        outputs = model(inputs)                                 # forward pass to get predictions from model
        loss = criterion(outputs, labels)                       # loss calculation compares predicted outputs and ground-truth labels

        loss.backward()                                         # back propogation to compute gradients
        optimizer.step()                                        # update model's weights and biases with gradients
        
        train_loss += loss.item()                               # update current mini-batch loss to running loss
        
        total_mini_batches = 50000 // batch_size                        # total # of mini-batches = # of training examples / batch size
        if batch_index % total_mini_batches == total_mini_batches - 1:  # print avg loss for current mini-batch, # of training examples / batch size = 50000/256 = 195
            avg_train_loss = train_loss / total_mini_batches            # avg loss calculation, divide by total # of mini-batches
            total_train_loss.append(avg_train_loss)             # update the total training loss list with the next avg training loss for the current mini-batch
            train_loss = 0.0                                    # reset running training loss
            return avg_train_loss                               # return avg training loss to be displayed

def test(model, criterion):
    '''
    Tests the model on test dataset

    Args:
    - model: the desired model for training
    - criterion: loss function (usually CrossEntropyLoss)

    Return:
    - (avg_test_loss, test_accuracy): tuple of average test loss and accuracy for the current epoch
    '''
    test_loss = 0.0                                             # tracks the test loss for the current epoch
    model.eval()                                                # enable testing mode on the model
    correct = 0
    total = 0

    with torch.no_grad():                                       # disable gradient calculations since we're not training
        for data in testloader:                                 # iterate over mini-batches in test dataset
            inputs, labels = data                               # assign input image and ground truth label for each mini-batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)                             # forward pass: test images sent through network for predictions
            loss = criterion(outputs, labels)

            test_loss += loss.item()                            # calculate test loss
            _, predicted = torch.max(outputs.data, 1)           # predicted class indices are obtained

            # updates total image count for each mini-batch
            # labels is the ground truth class labels 
            total += labels.size(0)
        
            # compares predicted class indices with ground truth class indicies element-wise (T/F 1/0), returns a Boolean tensor
            # .sum() calculates the number of correct predictions since 1 is true and 0 is false
            # .item() extracts numerical value of the summation (integer)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(testloader)                 # avg test loss calculation, divide by the size of the entire test set
    test_accuracy = 100 * correct / total                       # calculation of the model's test accuracy (%) on the entire test set

    total_test_loss.append(avg_test_loss)                       # update the total test loss list with the avg test loss for the current epoch
    total_test_accuracy.append(test_accuracy)                   # update the total test accuracy list 
    return avg_test_loss, test_accuracy                         # return tuple of avg test loss and test accuracy to be displayed

def reset_trackers():
    '''
    Resets lists that track training losses, test losses, and test accuracies

    Args: None
    
    Return: None
    '''
    global total_test_loss, total_train_loss, total_test_accuracy
    total_train_loss = []
    total_test_loss = []
    total_test_accuracy = []

def class_accuracy(model):
    '''
    Calculates the model's accuracy on classifying the test dataset
    Outputs the % accuracy for each of the 10 classes

    Args:
    - model: the model used to calculate class accuracies

    Return: None
    '''
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

print("--> Preparing data...")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # these normalization values are commonly used for CIFAR10 dataset
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform_test)

batch_size = 256
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("--> Initializing model...")

# NOTE: make changes to model, file_name, for different models
model = VGG('VGG16')
# model = MobileNet()
# model = MobileNetV2()
# model = ResNet18()

model_file_name = 'vgg16_trained.pth'   # .pth file name to save the trained model
LC_file_name = 'vgg16_trained_LC.png'   # .png file name to save the learning curve
LC_title = 'VGG16 Learning Curve'       # title of the learning curve
pretrained_model_exists = os.path.exists(os.path.join('Trained_Models', model_file_name))   # boolean flag to check whether a saved pretrained model exists
num_epochs = 60                         # number of training iterations

criterion = nn.CrossEntropyLoss()                                                                       # applies softmax   
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)                     # used to update parameters (weights and biases) to minimize loss function
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60, eta_min=0.001)    # scheduler used to adjust learning rate during training

model_complexity(model, device)     # obtain model complexity of the unpruned model

if not pretrained_model_exists:     # if the pretrained model does not exist: train the unpruned model
    print("--> Pretrained model not found.")

    model = model.to(device)                # moves model to device to ensure the next computations are performed on the specified device
    if device == torch.device('cuda:0'):
        cudnn.benchmark = True              # enables cuDNN benchmarking mode for best algorithm during convolutional operations

    # training and testing loop
    print("--> Training and testing in progress...")
    start_time = time.time()

    for epoch in range(num_epochs):
        avg_train_loss = train(model, criterion, optimizer, batch_size)
        test_data = test(model, criterion)
        avg_test_loss, test_accuracy = test_data
        scheduler.step()
        print(f'[Epoch {epoch + 1:>3}/{num_epochs}]\t Avg Train Loss: {avg_train_loss:.3f}\t Avg Test Loss: {avg_test_loss:.3f}\t Test Acc: {test_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Training and testing completed in {total_time:.2f} seconds.')

    save_model(model, model_file_name)
    class_accuracy(model)
    save_learning_curve(num_epochs, LC_file_name, LC_title)
else:   # else the pretrained model exists: load the trained (unpruned) model
    print("--> Pretrained model detected.")

    model.load_state_dict(torch.load(os.path.join('Trained_Models', model_file_name), map_location=device))
    model = model.to(device)
    if device == torch.device('cuda:0'):
        cudnn.benchmark = True

# loop to uniformly prune the model and retest the model at pruning ratio increments of 5%
reset_trackers()

for i in range(0,19):   # iterate over pruning ratios from 5% to 95%
    pruning_ratio = 0.05
    uniform_prune_vgg16(model, pruning_ratio)
    test_data = test(model, criterion)
    avg_test_loss, test_accuracy = test_data
    print(f'[Pruning Ratio: {0.05 + i * pruning_ratio:.2f}]\t Test Accuracy: {test_accuracy:.2f}%')

save_pruned_accuracy_curve('vgg16_L1norm_uniformly_pruned_AC.png', total_test_accuracy, 'VGG16 L1norm Uniformly Pruned Accuracy Curve')

'''
Prune the model and implement fine-tuning with original weights
'''
