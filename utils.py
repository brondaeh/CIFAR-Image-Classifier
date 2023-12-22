import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from ptflops import get_model_complexity_info

def imshow(img):
    '''
    Visualization of training images

    Args:
    - img: single pytorch tensor image

    Return: None
    '''
    print("--> Visualizing training images...")
    img = img / 2 + 0.5                             # unnormalize
    npimg = img.numpy()                             # convert to numpy array
    plt.imshow(np.transpose(npimg, (1, 2, 0)))      # change order of dimensions from pytorch (channels, height, wdith) -> plt (height, width, channels)
    plt.show()

def saveLearningCurve(num_epochs, file_name, total_train_loss, total_test_loss, title):
    '''
    Saves a learning curve figure: training loss and test loss over total epochs

    Args:
    - num_epochs: number of epochs
    - file_name: .png file name of the learning curve figure
    - total_train_loss: list that tracks the running loss for each epoch during training
    - total_test_loss: list that tracks the running loss for each epoch during testing
    - title: title of the figure

    Return: None
    '''
    plt.plot(range(num_epochs), total_train_loss, label='Training Loss')
    plt.plot(range(num_epochs), total_test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.ylim(0, 5)
    plt.xlim(0, num_epochs)

    folder_name = 'Learning_Curves'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()
    print(f"--> Learning curve saved to: {os.path.join(folder_name, file_name)}")

def modelComplexity(model, device):
    '''
    Calculates model complexity and number of parameters in the model
    
    Args:
    - model: the desired model for complexity calculation
    - device: the device used for calculation

    Return: None
    '''
    print ("--> Calculating model complexity...")
    if device == 'cuda:0':
        with torch.cuda.device(device):
            maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
            print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    else:
        maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
        print('{:<30}  {:<8}'.format('Number of Parameters: ', params))

def saveModel(model, file_name, folder_name):
    '''
    Saves the model to Trained_Models folder

    Args:
    - model: the model to save
    - file_name: the desired .pth file name for the model
    - folder_name: the folder to store the model file

    Return: None
    '''
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    torch.save(model.state_dict(), os.path.join(folder_name, file_name))
    print(f"--> Model saved to: {os.path.join(folder_name, file_name)}")

def savePrunedAccuracyCurve(file_name, total_test_accuracy, title):
    '''
    Saves an accuracy curve figure: accuracy (%) over pruning ratio (%)

    Args:
    - file_name: the desired .png file name for the accuracy curve
    - total_test_accuracy: list of test accuracies
    - title: title of the figure

    Return: None
    '''
    pruning_ratios = list(range(5, 100, 5))
    plt.plot(pruning_ratios, total_test_accuracy, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('Pruning Ratio (%)')
    plt.ylabel('Accuracy (%)')
    plt.xticks(pruning_ratios)
    plt.yticks(range(0, 101, 10))
    plt.grid()

    folder_name = 'Accuracy_Curves'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()
    print(f"--> Accuracy curve saved to: {os.path.join(folder_name, file_name)}")