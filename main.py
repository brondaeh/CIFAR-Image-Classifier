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
import matplotlib.pyplot as plt

from Models import *
from Pruner import *
from ptflops import get_model_complexity_info


'''
Data Preprocessing
------------------------------------------------------
Definitions of data augmentations and dataset loaders
'''
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


'''
Methods
------------------------------------------------------
'''
def imshow(img):
    '''
    Visualization of training images

    Args:
    - img: single pytorch tensor image

    Return: None
    '''
    img = img / 2 + 0.5                             # unnormalize
    npimg = img.numpy()                             # convert to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))      # change order of dimensions from pytorch (channels, height, wdith) -> plt (height, width, channels)
    plt.show()

total_train_loss = []   # list to track total training loss for learning curve
def train(model, criterion, optimizer):
    '''
    Trains the model with forward and backprop

    Args:
    - model: the desired model for training
    - criterion: loss function (usually CrossEntropyLoss)
    - optimizer: algorithm used to minimize loss function (usually SGD)

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

        loss.backward()                                         # backward pass to compute gradients
        optimizer.step()                                        # update model's weights and biases with gradients
        
        train_loss += loss.item()                               # update current mini-batch loss to running loss
        
        if batch_index % 195 == 194:                            # print avg loss for current mini-batch, # of training examples/batch size = 50000/256 = 195
            avg_train_loss = train_loss / 195                   # avg loss calculation, divide by total # of mini-batches
            total_train_loss.append(avg_train_loss)             # update the total training loss list with the next avg training loss for the current mini-batch
            train_loss = 0.0                                    # reset running training loss
            return avg_train_loss                               # return avg training loss to be displayed

total_test_loss = []    # list to track total test loss for learning curve
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
    model.eval()                                                  # enable testing mode on the model
    correct = 0
    total = 0

    with torch.no_grad():                                       # disable gradient calculations since we're not training
        for data in testloader:                                 # iterate over mini-batches in test dataset
            inputs, labels = data                               # assign input image and ground truth label for each mini-batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)                               # forward pass: test images sent through network for predictions
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
    test_accuracy = 100 * correct / total                       # calculation of the model's accuracy on the entire test set

    total_test_loss.append(avg_test_loss)                       # update the total test loss list with the avg test loss for the current epoch
    return avg_test_loss, test_accuracy                         # return tuple of avg test loss and test accuracy to be displayed

def reset_losses():
    '''
    Resets training and testing losses

    Args: None
    
    Return: None
    '''
    global total_test_loss, total_train_loss
    total_train_loss = []
    total_test_loss = []

def train_and_test(num_epochs, model):
    '''
    Trains and tests the model for num_epochs
    Displays the total training and testing time elapsed and outputs average losses for each epoch
    
    Args:
    - num_epochs: the number of iterations for training and testing
    - model: the desired model

    Return: None
    '''
    print("--> Training and testing in progress...")
    criterion = nn.CrossEntropyLoss()                                                                       # applies softmax   
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.001)                     # used to update parameters (weights and biases) to minimize loss function
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=60, eta_min=0.001)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        avg_train_loss = train(model, criterion, optimizer)
        test_data = test(model, criterion)
        avg_test_loss, test_accuracy = test_data
        scheduler.step()

        print(f'[Epoch {epoch + 1:>3}]\t Avg Training Loss: {avg_train_loss:.3f}\t Avg Test Loss: {avg_test_loss:.3f}\t Test Accuracy: {test_accuracy:.2f}%')

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Training and testing completed in {total_time:.2f} seconds.')

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

def save_learning_curve(num_epochs, file_name):
    '''
    Saves a learning curve figure with training loss and test loss over total epochs

    Args:
    - num_epochs: number of epochs
    - file_name: .png file name of the learning curve figure

    Return: None
    '''
    plt.plot(range(num_epochs), total_train_loss, label='Training Loss')
    plt.plot(range(num_epochs), total_test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid()
    plt.legend()

    folder_name = 'Learning_Curves'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(os.path.join(folder_name, file_name))

    # plt.show()

def model_complexity(model):
    '''
    Calculates model complexity and number of parameters in the model
    
    Args:
    - model: the desired model for complexity calculation

    Return: None
    '''
    print ("--> Calculating Model Complexity...")

    with torch.cuda.device(0):
        maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)

        print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
        print('{:<30}  {:<8}'.format('Number of Parameters: ', params))

def save_model(model, file_name):
    '''
    Saves the model to Trained_Models folder

    Args:
    - model: the model to save
    - filename: the desired .pth file name for the model

    Return: None
    '''
    folder_name = 'Trained_Models'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    torch.save(model.state_dict(), os.path.join(folder_name, file_name))

def prune(model):
    '''
    Prunes the model

    Args:
    - model: the model used for pruning

    Return: None
    '''
    print ("--> Pruning model...")
    pruner = pruning_engine(pruning_method='L1norm', individual=True)       # define L1norm pruning method

    pruned_layer = model.features[0]
    pruner.set_pruning_ratio(0.2)
    pruner.set_layer(pruned_layer,main_layer=True)
    remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
    model.features[0] = pruner.remove_filter_by_index(remove_filter_idx)

    pruned_layer = model.features[1]
    pruner.set_pruning_ratio(0.2)
    pruner.set_layer(pruned_layer)
    remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
    model.features[1] = pruner.remove_Bn(remove_filter_idx)

    pruned_layer = model.features[3]
    pruner.set_pruning_ratio(0.2)
    pruner.set_layer(pruned_layer)
    remove_filter_idx = pruner.get_remove_filter_idx()["current_layer"]
    model.features[3] = pruner.remove_kernel_by_index(remove_filter_idx)


'''
Training, Testing, and Pruning
------------------------------------------------------
'''
print("--> Initializing model...")

# NOTE: change net for different models
model = VGG('VGG16')
# model = MobileNet()
# model = MobileNetV2()
# model = ResNet18()

model_complexity(model)                         # obtain model complexity of the unpruned model

file_name = 'vgg16_trained.pth'
pretrained_model_exists = os.path.exists(os.path.join('Trained_Models', file_name))

num_epochs = 60                                 # number of training and testing iterations

# if the pretrained model does not exist: train the unpruned model -> prune the model -> retrain the pruned model
if not pretrained_model_exists:
    print("--> Pretrained model not found.")

    model = model.to(device)                        # moves model to device to ensure the next computations are performed on the specified device
    if device == torch.device('cuda:0'):
        # net = torch.nn.DataParallel(net)          # wraps model with DataParallel to parallelize training process on GPUs if available
        cudnn.benchmark = True                      # enables cuDNN benchmarking mode for best algorithm during convolutional operations

    # training and testing
    train_and_test(num_epochs, model)               # train and test the unpruned model
    save_model(model, file_name)                    # save the trained (unpruned) model
    class_accuracy(model)                           # obtain accuracy across all classes
    save_learning_curve(num_epochs, 'vgg16_trained_LC.png')

    # pruning
    prune(model)
    model_complexity(model)

    # retraining and retesting
    reset_losses()
    train_and_test(num_epochs, model)
    save_model(model, 'vgg16_pruned_trained.pth')
    class_accuracy(model)
    save_learning_curve(num_epochs, 'vgg16_pruned_trained_LC.png')

# else the pretrained model exists: prune the model -> retrain the pruned model
else: 
    print("--> Pretrained model detected.")

    model.load_state_dict(torch.load('Trained_Models/vgg16_trained.pth', map_location=device))
    model = model.to(device)
    if device == torch.device('cuda:0'):
        cudnn.benchmark = True

    # pruning
    prune(model)
    model_complexity(model)

    # retraining and retesting
    reset_losses()
    train_and_test(num_epochs, model)
    save_model(model, 'vgg16_pruned_trained.pth')
    class_accuracy(model)
    save_learning_curve(num_epochs, 'vgg16_pruned_trained_LC.png')    
