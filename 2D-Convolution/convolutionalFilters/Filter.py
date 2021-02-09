import matplotlib.pyplot as plt
from skimage import io
from scipy.signal import convolve
import numpy as np


class Filter():
    def __init__(self, img, kernel, title):
        self.img = img
        self.kernel = kernel
        self.result = []
        self.title = title

    def plotResult(self, absolute=False):
        _, axs = plt.subplots(1, 3, figsize=(15, 5))

        try:
            plot = [self.img, self.kernel]

            if absolute:
                plot.append(np.abs(self.result))
            else:
                plot.append(self.result)

        except Exception as e:
            print('Maybe you forgot to convolve the image and filter!', e)
            return

        titles = ['Imagem',
                  self.title,
                  'Mapa de Ativação (Absoluto)' if absolute else 'Mapa de Ativação']

        for k, ax in enumerate(axs):
            ax.imshow(plot[k], cmap='gray')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title(titles[k])

        for i, line in enumerate(self.kernel):
            for j, col in enumerate(line):
                axs[1].text(j, i, '{:.2f}'.format(col),
                            fontsize=12,
                            color='red',
                            ha='center',
                            va='center')

        plt.show()

    def convolveFilter(self):
        self.result = convolve(self.img, self.kernel, mode='valid')
