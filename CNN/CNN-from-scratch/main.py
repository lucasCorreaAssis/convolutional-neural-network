# Implementation
import torch
from torch import nn

# Data load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

# Implementing LeNet
'''
    LeNet parameters

    |----------------------------------------------------------------------|
    |                     | Feature |       | Kernel |        |            |
    |       Layer         |   map   | Size  |  size  | Stride | Activation |
    |----------------------------------------------------------------------|
    | Input |    Image    |    1    | 32x32 |    -   |    -   |     -      |
    |----------------------------------------------------------------------|
    |   1   | Convolution |    1    | 28x28 |   5x5  |    1   |    tanh    |
    |----------------------------------------------------------------------|
    |       |   Average   |         |       |        |        |            |
    |   2   |   Pooling   |    6    | 14x14 |   2x2  |    2   |    tanh    |
    |----------------------------------------------------------------------|
    |   3   | Convolution |    16   | 10x10 |   5x5  |    1   |    tanh    |
    |----------------------------------------------------------------------|
    |       |   Average   |         |       |        |        |            |
    |   4   |   Pooling   |    16   |  5x5  |   2x2  |    2   |    tanh    |
    |----------------------------------------------------------------------|
    |   5   | Convolution |    120  |  1x1  |   5x5  |    1   |    tanh    |
    |----------------------------------------------------------------------|
    |   6   |      FC     |    -    |   84  |    -   |    -   |    tanh    |
    |----------------------------------------------------------------------|
    | Output|      FC     |    -    |   10  |    -   |    -   |   softmax  |
    |----------------------------------------------------------------------|
'''

lenet = nn.Sequential(
                      # input : (b, 3, 32, 32)
                      # output : (b, 6, 28, 28)
                      nn.Conv2d(3, 6, kernel_size=5),
                      nn.BatchNorm2d(6),
                      nn.Tanh(),
                      # input : (b, 3, 28, 28)
                      # output : (b, 6, 14, 14)
                      nn.AvgPool2d(kernel_size=2),

                      # input : (b, 6, 14, 14)
                      # output : (b, 16, 10, 10)
                      nn.Conv2d(6, 16, kernel_size=5),
                      nn.BatchNorm2d(16),
                      nn.Tanh(),
                      # input : (b, 16, 10, 10)
                      # output : (b, 16, 5, 5)
                      nn.AvgPool2d(kernel_size=2),

                      # input : (b, 16, 5, 5)
                      # output : (b, 120, 1, 1)
                      nn.Conv2d(16, 120, kernel_size=5),
                      nn.BatchNorm2d(120),
                      nn.Tanh(),

                      # input : (b, 120, 1, 1)
                      # output : (b, 120*1*1)
                      nn.Flatten(),

                      # input : (b, 120)
                      # output : (b, 84)
                      nn.Linear(120, 84),
                      nn.Tanh(),

                      # input : (b, 84)
                      # output : (b, 10)
                      nn.Linear(84, 10),
                      nn.Softmax())

print(lenet)
