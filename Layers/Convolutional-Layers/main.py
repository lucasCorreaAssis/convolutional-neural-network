import torch
from torch import nn

from skimage import data
import matplotlib.pyplot as plt


# 1 channel sample
brick = data.brick()

# 3 channels sample
astronaut = data.astronaut()

# Plot images
fig, axs = plt.subplots(2)
fig.suptitle('Images samples')
axs[0].imshow(brick, cmap='gray')
axs[1].imshow(astronaut)
plt.show()

# We'll define a convolutional layer to do the forward in the image brick
# Requirement:
#   * The input must be a tensor
#   * The convolutional layer expects a input with the following dimensions:
#       * BATCH x CHANNELS x HEIGHT x WIDTH

# 1 CHANNEL CONVOLUTION

conv = nn.Conv2d(in_channels=1,
                 out_channels=16,
                 kernel_size=3,
                 padding=1)

brick_tensor = torch.Tensor(brick)
print(brick_tensor.size())
brick_tensor = brick_tensor.view(1,
                                 1,
                                 brick_tensor.size(0),
                                 brick_tensor.size(1))

print(brick_tensor.size())

activation_map = conv(brick_tensor)
print(activation_map.size())

# 3 CHANNELS CONVOLUTION

conv = nn.Conv2d(in_channels=3,
                 out_channels=16,
                 kernel_size=3,
                 padding=1)

astrounaut_tensor = torch.Tensor(astronaut)
print(astrounaut_tensor.size())
astrounaut_tensor = astrounaut_tensor.view(1,
                                           3,
                                           astrounaut_tensor.size(0),
                                           astrounaut_tensor.size(1))

print(astrounaut_tensor.size())

activation_map = conv(astrounaut_tensor)
print(activation_map.size())
