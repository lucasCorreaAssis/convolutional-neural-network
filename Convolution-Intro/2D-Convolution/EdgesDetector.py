from scipy.signal import convolve
import matplotlib.pyplot as plt
import numpy as np


class EdgesDetector():
    '''Edges Detector based on 2D convolution'''
    def __init__(self, img):
        self.img = img
        self.feature_map = []
        self.props = {'feature_map_title': ''}
        self.vertical_kernel = [[1, 0, -1],
                                [1, 0, -1],
                                [1, 0, -1]]

        self.horizontal_kernel = [[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]]

    def convolveImg(self, **kwargs):
        if 'kernel' not in kwargs:
            print('invalid kernel! (edgeDetector.convolve(kernel=...))')
            return

        if kwargs['kernel'] == 'vertical':
            self.props['feature_map_title'] = 'vertical'
            self.feature_map = convolve(self.img,
                                        self.vertical_kernel,
                                        mode='valid')
        elif kwargs['kernel'] == 'horizontal':
            self.props['feature_map_title'] = 'horizontal'
            self.feature_map = convolve(self.img,
                                        self.horizontal_kernel,
                                        mode='valid')
        else:
            print('invalid kernel [horizontal, vertical]')
            return

    def plotFeatureMap(self):
        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(1, 2, 1)
        plt.imshow(self.img, cmap='Greys')
        plt.title('original image')
        fig.add_subplot(1, 2, 2)
        plt.title('feature map ({})'.format(self.props['feature_map_title']))
        plt.imshow(self.feature_map, cmap='Greys')
        plt.show()

    def plotVerticalKernel(self):
        self.plotKernel(self.vertical_kernel, 'Vertical Kernel')

    def plotHorizontalKernel(self):
        self.plotKernel(self.horizontal_kernel, 'Horizontal Kernel')

    def plotKernel(self, values, title):
        plt.figure(figsize=(len(values), len(values)))
        plt.imshow(values, cmap='gray')
        for i, line in enumerate(values):
            for j, col in enumerate(line):
                plt.text(j, i, '{:.0f}'.format(col),
                         fontsize=16,
                         color='red',
                         ha='center',
                         va='center')
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(title+'.png', format='png', dpi=100, bbox_inches='tight')
        plt.show()