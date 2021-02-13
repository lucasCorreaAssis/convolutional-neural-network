from skimage import io
import numpy as np
from Filter import Filter


'''We'll implement the sobel filter'''


def getSobelKernel(**kwargs):
    if 'direction' not in kwargs:
        print('You have to declare the direction of the filter')
        return

    sobel = np.zeros((3, 3))
    if kwargs['direction'] == 'vertical':
        sobel[:, 0] = -1
        sobel[:, 2] = 1
        sobel[1, [0, 2]] = [-2, 2]
    elif kwargs['direction'] == 'horizontal':
        sobel[0, :] = -1
        sobel[2, :] = 1
        sobel[[0, 2], 1] = [-2, 2]
    else:
        print('incorrect direction')
        return

    return sobel


img = io.imread('GUI.jpg')
img = img[:, :, 0]

sobelKernel = getSobelKernel(direction='vertical')
filter = Filter(img, sobelKernel, 'Sobel Vertical Kernel')
filter.convolveFilter()
filter.plotResult()
filter.plotResult(absolute=True)

sobelKernel = getSobelKernel(direction='horizontal')
filter = Filter(img, sobelKernel, 'Sobel Horizontal Kernel')
filter.convolveFilter()
filter.plotResult()
filter.plotResult(absolute=True)
