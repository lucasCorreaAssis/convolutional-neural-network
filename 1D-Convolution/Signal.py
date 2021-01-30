import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve


class Signal:
    def __init__(self, signal, title, **kwargs):
        if 'domain' in kwargs:
            self.domain = kwargs['domain']
        else:
            self.domain = np.linspace(0, len(signal), len(signal))
        self.signal = signal
        self.title = title

        self.props = {
            'shifted': False,
            'shifted_factor': 0,
            'inverted': False,
        }

    def plot(self):
        plt.figure(figsize=(12, 3))
        plt.plot(self.domain, self.signal)
        plt.show()

    def plotAsArray(self):
        plt.figure(figsize=(len(self.signal), 2))
        plt.imshow(self.signal[np.newaxis, :], cmap='gray')

        for k, s in enumerate(self.signal):
            plt.text(k, 0, '{:.1f}'.format(s),
                     fontsize=16,
                     color='red',
                     ha='center',
                     va='center')

        plt.title(self.title, fontsize=18)
        plt.yticks([])
        plt.show()

    def resetSignal(self):
        if self.props['inverted']:
            self.signal = np.flip(self.signal)

        if self.props['shifted']:
            self.signal = np.delete(self.signal,
                                    [i for i in range(0, self.props['shifted_factor'])])

            self.props['shifted_factor'] = 0

        self.props['inverted'] = False
        self.props['shifted'] = False

    def shift(self, factor):
        if self.props['shifted']:
            self.resetSignal()

        self.props['shifted_factor'] = factor
        displacement = [float('nan')] * factor
        self.signal = np.hstack((displacement, self.signal))
        self.props['shifted'] = True

    def invert(self):
        if self.props['inverted']:
            return

        self.signal = np.flip(self.signal)
        self.props['inverted'] = True
