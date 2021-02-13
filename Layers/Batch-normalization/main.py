import torch
from torch import nn
from skimage import data


def getAustronautTensor():
    '''Returns a 3 channel image of an astronaut'''
    astronaut = data.astronaut()
    astrounaut_tensor = torch.Tensor(astronaut)
    astrounaut_tensor = astrounaut_tensor.view(1,
                                               3,
                                               astrounaut_tensor.size(0),
                                               astrounaut_tensor.size(1))

    return astrounaut_tensor


conv_block = nn.Sequential(
                           nn.Conv2d(3, 32, kernel_size=3, padding=1),
                           nn.BatchNorm2d(32),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=10)
            )

print(conv_block)
austronaut_tensor = getAustronautTensor()
print(austronaut_tensor.size())

output = conv_block(austronaut_tensor)
print(output.size())
