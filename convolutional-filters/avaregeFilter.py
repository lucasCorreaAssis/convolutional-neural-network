from skimage import io
import numpy as np
from Filter import Filter


'''We'll implement the avarege filter'''


def getAvaregeKernel(size):
    avarege_kernel = np.zeros((size, size))
    avarege_kernel[:] = 1.0 / (size**2)

    return avarege_kernel


img = io.imread('GUI.jpg')
img = img[:, :, 0]

avaregeKernel = getAvaregeKernel(3)
filter = Filter(img, avaregeKernel, 'Avarege Kernel')
filter.convolveFilter()
filter.plotResult()
filter.plotResult(absolute=True)
