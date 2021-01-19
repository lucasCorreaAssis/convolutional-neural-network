import matplotlib.pyplot as plt 
from matplotlib import patches
import numpy as np

import torch
from torchvision import datasets, transforms

class MNIST:
    '''MNIST possui 10 classes, os digitos entre 0 e 9!'''
    def __init__(self):
        self.mnistDataset = datasets.MNIST('./dataset',
                               train = False,
                               transform = transforms.ToTensor(),
                               download = False)

    def printDatasetInfo(self):
        '''Print data and target types, data size and target value'''
        dado, rotulo = self.mnistDataset[0]
        print(type(dado), type(rotulo))
        print(dado.size(), rotulo)

    def plotDatasetSamples(self):
        '''Plot 10 first samples from the dataset'''
        fig, axs = plt.subplots(1, 10, figsize=(15,4))
        for i in range(10):
            dado, rotulo = self.mnistDataset[i]
            axs[i].imshow(dado[0], cmap='gray')
            axs[i].set_title(str(rotulo))
        plt.show()