import torch
from torch import nn

from skimage import data

tensor = torch.FloatTensor([[
                             [1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]]])

pool = nn.MaxPool2d(2, stride=1)

output = pool(tensor)

print(tensor.size())
print(tensor)
print(output.size())
print(output)
