import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

import torch 
from torchvision import datasets, transforms
import torchvision

class VOC:
    def __init__(self):
        self.vocDataset = torchvision.datasets.VOCDetection('./dataset',
                                                     image_set = 'train',
                                                     download = False,
                                                     transform = transforms.ToTensor())
        
        self.dataSample, self.targetSample = self.vocDataset[0]
        self.dataSample = self.organizeDataDimensions(self.dataSample)
    
    def printDatasetInfo(self):
        '''Print data and target types and data size'''
        print(type(self.dataSample), type(self.targetSample))
        print(self.dataSample.size())

    def organizeDataDimensions(self, data):
        '''Convert image from format first channel to last channel'''
        return data.permute(1, 2, 0)
    
    def plotDataSetSample(self):
        '''Plot sample image from VOC dataset'''
        plt.figure(figsize=(8,8))
        plt.imshow(self.dataSample)
        plt.show()
    
    def printTargetSample(self):
        '''Print a target to show its structure'''
        print(self.targetSample)

    def plotSampleBoundingBox(self):
        '''plot sample target'''
        bbox = self.targetSample['annotation']['object'][0]['bndbox']
        xmax = int(bbox['xmax'])
        xmin = int(bbox['xmin'])
        ymax = int(bbox['ymax'])
        ymin = int(bbox['ymin'])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(self.dataSample)

        width, height = xmax-xmin, ymax-ymin
        rect = patches.Rectangle((xmin, ymin), width, height, 
                                  fill=False, 
                                  color='r', 
                                  linewidth=4)

        ax.add_patch(rect)
        plt.show()
