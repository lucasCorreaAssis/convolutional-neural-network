from skimage import io
import numpy as np
from Filter import Filter


'''We'll implement the laplace filter'''


def getLaplaceKernel(size):
    laplace_kernel = np.ones((size, size)) * -1
    laplace_kernel[1, 1] = 8

    return laplace_kernel


img = io.imread('GUI.jpg')
img = img[:, :, 0]

laplaceKernel = getLaplaceKernel(3)
filter = Filter(img, laplaceKernel, 'Laplace Kernel')
filter.convolveFilter()
filter.plotResult()
filter.plotResult(absolute=True)
