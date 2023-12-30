'''Training and pruning neural network models on the CIFAR10 and CIFAR100 dataset in PyTorch'''

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
import yaml

from Models import *
from Pruner import *
from Pruning_Functions import *
from utils import saveLearningCurve, modelComplexity, saveModel, savePrunedAccuracyCurve

# Create lists to track the total train loss, test loss, and test accuracies
total_train_loss = []
total_test_loss = []
total_test_accuracy = []

# Specify the device to use; if GPU not available then use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Chosen device: {device}')

# Read the config.yaml file to load training parameters
with open('config.yaml', 'r') as f:
    train_config = yaml.load(f, yaml.FullLoader)['Train_Config']

num_epochs = train_config['num_epochs']
batch_size = train_config['batch_size']
learning_rate = train_config['learning_rate']
momentum = train_config['momentum']
weight_decay = train_config['weight_decay']
dataset = train_config['dataset']
pruning_flag = train_config['pruning_flag']
model_folder_name = train_config['model_folder_name']
model_file_name = train_config['model_file_name']
LC_file_name = train_config['LC_file_name']
LC_title = train_config['LC_title']

# Choose dataset mean and std based on the chosen dataset
if dataset == 'CIFAR10':
    dataset_mean = [0.4914, 0.4822, 0.4465]
    dataset_std = [0.2470, 0.2435, 0.2616]
    input_size = 32
    num_classes = 10
elif dataset == 'CIFAR100':
    dataset_mean = [0.5071, 0.4867, 0.4408]
    dataset_std = [0.2675, 0.2565, 0.2761]
    input_size = 32
    num_classes = 100
else:
    raise ValueError('Invalid dataset name. Please specify either CIFAR10 or CIFAR100 in config.yaml.')

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

# Load the dataset based on the chosen dataset
if dataset == 'CIFAR10':
    trainset = torchvision.datasets.CIFAR10(root='./Data', train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR10(root='./Data', train=False, transform=transform_test, download=True)
elif dataset == 'CIFAR100':
    trainset = torchvision.datasets.CIFAR100(root='./Data', train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR100(root='./Data', train=False, transform=transform_test, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
classes = trainset.classes

print('--> Preparing model...')
if train_config['model'] == 'VGG16':
    model = VGG('VGG16', num_classes=num_classes)
    pruned_model = VGG('VGG16',num_classes=num_classes)
elif train_config['model'] == 'MobileNetV1':
    model = MobileNet(num_classes=num_classes)
    pruned_model = MobileNet(num_classes=num_classes)
elif train_config['model'] == 'MobileNetV2':
    model = MobileNetV2(num_classes=num_classes)
    pruned_model = MobileNetV2(num_classes=num_classes)
elif train_config['model'] == 'ResNet18':
    model = ResNet18(num_classes=num_classes)
    pruned_model = ResNet18(num_classes=num_classes)
else:
    raise ValueError('Invalid model name. Please specify the model name in config.yaml.')

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
    train_loss = 0.0        # tracks the running train loss for each epoch
    model.train()           # enable training mode on the model

    for batch_index, data in enumerate(trainloader, 0):                 # iterate thru mini-batches, batch_index = current mini-batch, data = trainloader data
        inputs, labels = data                                           # unpack mini-batch data to input image and corresponding ground-truth labels [inputs, labels]
        inputs, labels = inputs.to(device), labels.to(device)           # send inputs and labels to specified device for computations

        optimizer.zero_grad()                                           # zero all parameter gradients
        outputs = model(inputs)                                         # forward pass to get predictions from model
        loss = criterion(outputs, labels)                               # loss calculation compares predicted outputs and ground-truth labels

        loss.backward()                                                 # back propogation to compute gradients
        optimizer.step()                                                # update model's weights and biases with gradients
        
        train_loss += loss.item()                                       # update current mini-batch loss to running loss
        
        total_train_examples = 50000                                    # 50000 training examples in cifar10 and cifar100
        total_mini_batches = total_train_examples // batch_size         # total # of mini-batches = # of training examples / batch size
        if batch_index % total_mini_batches == total_mini_batches - 1:  # print avg loss for current mini-batch, # of training examples / batch size = 50000/256 = 195
            avg_train_loss = train_loss / total_mini_batches            # avg loss calculation, divide by total # of mini-batches
            total_train_loss.append(avg_train_loss)                     # update the total training loss list with the next avg training loss for the current mini-batch
            train_loss = 0.0                                            # reset running training loss
            return avg_train_loss                                       # return avg training loss to be displayed

def test(model, criterion):
    '''
    Tests the model on test dataset

    Args:
    - model: the desired model for training
    - criterion: loss function (usually CrossEntropyLoss)

    Return:
    - (avg_test_loss, test_accuracy): tuple of average test loss and accuracy for the current epoch
    '''
    test_loss = 0.0         # tracks the test loss for the current epoch
    model.eval()            # enable testing mode on the model
    correct = 0
    total = 0

    with torch.no_grad():                                           # disable gradient calculations during testing
        for data in testloader:                                     # iterate over mini-batches in test dataset
            inputs, labels = data                                   # assign input image and ground truth label for each mini-batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)                                 # forward pass: test images sent through network for predictions
            loss = criterion(outputs, labels)

            test_loss += loss.item()                                # calculate test loss
            _, predicted = torch.max(outputs.data, 1)               # predicted class indices are obtained

            # updates total image count for each mini-batch
            # labels = the ground truth class labels 
            total += labels.size(0)
        
            # compares predicted class indices with ground truth class indicies element-wise (T/F 1/0), returns a Boolean tensor
            # .sum() calculates the number of correct predictions since 1 is true and 0 is false
            # .item() extracts numerical value of the summation (integer)
            correct += (predicted == labels).sum().item()

    avg_test_loss = test_loss / len(testloader)                     # avg test loss calculation, divide by the size of the entire test set
    test_accuracy = 100 * correct / total                           # calculation of the model's test accuracy (%) on the entire test set

    total_test_loss.append(avg_test_loss)                           # update the total test loss list with the avg test loss for the current epoch
    total_test_accuracy.append(test_accuracy)                       # update the total test accuracy list 
    return avg_test_loss, test_accuracy                             # return tuple of avg test loss and test accuracy to be displayed

def resetTrackers():
    '''
    Resets lists that track training losses, test losses, and test accuracies

    Args: None
    
    Return: None
    '''
    global total_test_loss, total_train_loss, total_test_accuracy
    total_train_loss = []
    total_test_loss = []
    total_test_accuracy = []

def classAccuracy(model):
    '''
    Calculates the model's accuracy on classifying the test dataset
    Outputs the % accuracy for each of the 10 classes and the top 1 accuracy

    Args:
    - model: the model used to calculate class accuracies

    Return:
    - overall_accuracy: the top1 accuracy for the model
    '''
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    total_samples = 0
    correct_samples = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            for label, prediction in zip(labels, predictions):
                total_samples += 1
                if label == prediction:
                    correct_samples += 1
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

    overall_accuracy = (correct_samples / total_samples) * 100.0
    print(f'Top1 accuracy: {overall_accuracy:.2f}%')

    return overall_accuracy

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)                                                        # criterion for classification tasks; applies softmax   
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)   # optimizer used to update parameters (weights and biases) to minimize loss function
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)               # scheduler used to adjust learning rate during training

modelComplexity(model, device)
pretrained_model_exists = os.path.exists(os.path.join(model_folder_name, model_file_name))      # boolean flag to check whether a saved pretrained model exists

# If the pretrained model is not found -> train the model (unpruned)
if not pretrained_model_exists:
    print('--> Pretrained model not found.')
    model = model.to(device)
    if device == torch.device('cuda:0'): cudnn.benchmark = True

    # Training and testing loop
    print('--> Training and testing in progress...')
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

    saveModel(model, model_file_name, model_folder_name)
    saveLearningCurve(num_epochs, LC_file_name, total_train_loss, total_test_loss, LC_title)
    top1_acc = classAccuracy(model)
# Else the pretrained model exists -> load the pretrained model (unpruned)
else:
    print('--> Pretrained model detected.')
    model.load_state_dict(torch.load(os.path.join('Trained_Models', model_file_name), map_location=device))
    model = model.to(device)
    if device == torch.device('cuda:0'): cudnn.benchmark = True
    top1_acc = classAccuracy(model)

# Stop point if the model will not be pruned
if not pruning_flag:
    raise Exception("Stop point: pruning will not be executed.")

# Read the config.yaml file to load pruning parameters
with open('config.yaml', 'r') as f:
    prune_config = yaml.load(f, yaml.FullLoader)['Prune_Config']

desired_pruning_ratio = prune_config['desired_pruning_ratio']
num_epochs = prune_config['num_epochs']
batch_size = prune_config['batch_size']
learning_rate = prune_config['learning_rate']
momentum = prune_config['momentum']
weight_decay = prune_config['weight_decay']
pruned_model_file_name = prune_config['pruned_model_file_name'].format(desired_pruning_ratio=desired_pruning_ratio)
AC_file_name = prune_config['AC_file_name']
AC_title = prune_config['AC_title']
pruned_LC_file_name = prune_config['pruned_LC_file_name'].format(desired_pruning_ratio=desired_pruning_ratio)
pruned_LC_title = prune_config['pruned_LC_title'].format(desired_pruning_ratio=desired_pruning_ratio)
pruned_model_exists = os.path.exists(os.path.join(model_folder_name, pruned_model_file_name))      # boolean flag to check whether a saved pruned model exists

resetTrackers()
total_test_accuracy.append(top1_acc)

# If a pruned model is not found -> prune the model for pruning ratios for 5% to 95% in increments of 5%
if not pruned_model_exists:
    print('--> Uniformly pruning the model...')
    for ratio in np.arange(5, 100, 5).tolist():
        if train_config['model'] == 'VGG16':            # create a new instance of the same model architecture called model_copy
            model_copy = VGG('VGG16', num_classes=num_classes)
        elif train_config['model'] == 'MobileNetV1':
            model_copy = MobileNet(num_classes=num_classes)
        elif train_config['model'] == 'MobileNetV2':
            model_copy = MobileNetV2(num_classes=num_classes)
        elif train_config['model'] == 'ResNet18':
            model_copy = ResNet18(num_classes=num_classes)
        else:
            raise ValueError('Invalid model name.')
        model_copy.load_state_dict(model.state_dict())  # load weights of the pretrained unpruned model to model_copy
        model_copy = model_copy.to(device)
        if device == torch.device('cuda:0'): cudnn.benchmark = True

        pruning_ratio = ratio / 100
        uniformPruneVGG16(model_copy, pruning_ratio)    # prune the copied model
        test_data = test(model_copy, criterion)
        avg_test_loss, test_accuracy = test_data
        print(f'[Pruning Ratio: {ratio}%]\t Test Accuracy: {test_accuracy:.2f}%')

        if ratio == desired_pruning_ratio:              # save pruned model only for the desired pruning ratio
            saveModel(model_copy, pruned_model_file_name, model_folder_name)

    savePrunedAccuracyCurve(AC_file_name, total_test_accuracy, AC_title)
else:
    print('--> Pruned model detected.')

# Fine-tune the pruned model by first loading the pruned model path to a model of the same pruned architecture
pruned_model = pruned_model.to(device)
if device == torch.device('cuda:0'): cudnn.benchmark = True
uniformPruneVGG16(pruned_model, desired_pruning_ratio / 100)
pruned_model.load_state_dict(torch.load(os.path.join('Trained_Models', pruned_model_file_name)))
modelComplexity(pruned_model, device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs)

resetTrackers()
print("--> Fine-tuning in progress...")
start_time = time.time()
for epoch in range(num_epochs):
    avg_train_loss = train(pruned_model, criterion, optimizer, batch_size)
    test_data = test(pruned_model, criterion)
    avg_test_loss, test_accuracy = test_data
    scheduler.step()
    print(f'[Epoch {epoch + 1:>3}/{num_epochs}]\t Avg Train Loss: {avg_train_loss:.3f}\t Avg Test Loss: {avg_test_loss:.3f}\t Test Acc: {test_accuracy:.2f}%')
end_time = time.time()
total_time = end_time - start_time
print(f'Fine-tuning completed in {total_time:.2f} seconds.')

saveModel(pruned_model, pruned_model_file_name, model_folder_name)
saveLearningCurve(num_epochs, pruned_LC_file_name, total_train_loss, total_test_loss, pruned_LC_title)
top1_acc = classAccuracy(pruned_model)
