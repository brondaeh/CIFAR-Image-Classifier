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

def save_learning_curve(num_epochs, file_name, total_train_loss, total_test_loss):
    '''
    Saves a learning curve figure: training loss and test loss over total epochs

    Args:
    - num_epochs: number of epochs
    - file_name: .png file name of the learning curve figure
    - total_train_loss: list that tracks the running loss for each epoch during training
    - total_test_loss: list that tracks the running loss for each epoch during testing

    Return: None
    '''
    plt.plot(range(num_epochs), total_train_loss, label='Training Loss')
    plt.plot(range(num_epochs), total_test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.grid()
    plt.legend()
    # plt.show()

    folder_name = 'Learning_Curves'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(os.path.join(folder_name, file_name))
    plt.close()

    print(f"--> Learning curve saved to: {os.path.join(folder_name, file_name)}")

def model_complexity(model, device):
    '''
    Calculates model complexity and number of parameters in the model
    
    Args:
    - model: the desired model for complexity calculation
    - device: the device used for calculation

    Return: None
    '''
    print ("--> Calculating Model Complexity...")

    if device == 'cuda:0':
        with torch.cuda.device(device):
            maccs, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True)

            print('{:<30}  {:<8}'.format('Computational Complexity: ', maccs))
            print('{:<30}  {:<8}'.format('Number of Parameters: ', params))
    else:
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

    print(f"--> Model saved to: {os.path.join(folder_name, file_name)}")

def save_accuracy_curve(file_name, ):
    '''
    Saves an accuracy curve figure: accuracy (%) over pruning ratio

    Args:
    - 
    '''
    pass