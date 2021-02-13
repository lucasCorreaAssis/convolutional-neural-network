# Implementation
import torch
from torch import nn, optim

# Data load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# Plots e análises
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import time, os

from network import getLeNet
from dataLoader import CIFAR10Loader
from train import train
from validate import validate

# Hyperparameters
args = {
    'epoch_num': 100,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'device': 'cpu',
    'batch_size': 50
}


def main():
    selectDevice()
    print(args['device'])

    lenet = getLeNet()
    lenet.to(args['device'])
    print(lenet)

    cifar_loader = CIFAR10Loader()
    train_loader = cifar_loader.getTrainLoader(args['batch_size'])
    test_loader = cifar_loader.getTestLoader(args['batch_size'])

    criterion = nn.CrossEntropyLoss().to(args['device'])
    optimizer = optim.Adam(lenet.parameters(),
                           lr=args['lr'],
                           weight_decay=args['weight_decay'])

    train_losses, test_losses = [], []
    for epoch in range(args['epoch_num']):
        # Train
        train_losses.append(train(train_loader,
                                  lenet,
                                  epoch,
                                  criterion,
                                  optimizer,
                                  **args))

        # Validate
        test_losses.append(validate(test_loader,
                                    lenet,
                                    epoch,
                                    criterion,
                                    optimizer,
                                    **args))


def selectDevice():
    '''Select cuda as device if available'''
    if torch.cuda.is_available():
        args['device'] = torch.device('cuda')
    else:
        args['device'] = torch.device('cpu')


if __name__ == "__main__":
    main()
