import matplotlib.pyplot as plt 
from matplotlib import patches
import numpy as np

import torch
from torchvision import datasets, transforms

class MNIST:
    '''MNIST have 10 classes (digits from 0 to 9)'''
    def __init__(self):
        self.mnistDataset = datasets.MNIST('./dataset',
                               train = False,
                               transform = transforms.ToTensor(),
                               download = False)

    def printDatasetInfo(self):
        '''Print data and target types, data size and target value'''
        data, target = self.mnistDataset[0]
        print(type(data), type(target))
        print(data.size(), target)

    def plotDatasetSamples(self):
        '''Plot 10 first samples from the dataset'''
        fig, axs = plt.subplots(1, 10, figsize=(15,4))
        for i in range(10):
            data, target = self.mnistDataset[i]
            axs[i].imshow(data[0], cmap='gray')
            axs[i].set_title(str(target))
        plt.show()