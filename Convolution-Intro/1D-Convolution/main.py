from Signal import Signal
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve


def getSinusoid():
    domain = np.linspace(0, 100, 100)
    kwargs = {'domain': domain}
    sin = 10 * np.sin(domain) * np.random.rand(domain.shape[0])
    sinusoid = Signal(sin, 'Sinusoid', **kwargs)

    return sinusoid


def getShortedSinusoid():
    domain = np.linspace(0, 10, 10)
    kwargs = {'domain': domain}
    sin = 10 * np.sin(domain) * np.random.rand(domain.shape[0])
    shortedSinusoid = Signal(sin, 'Shorted Sinusoid', **kwargs)

    return shortedSinusoid


def getKernel():
    kernel = Signal(np.asarray([1, 0, -1]), 'Kernel')

    return kernel


def plotSignalAndActivation(signal, activation):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, color='k', linewidth=4)
    plt.imshow(activation[np.newaxis, :],
               cmap='Reds',
               aspect='auto',
               alpha=0.8,
               extent=(0.5, 8.5, -10, 10))
    plt.colorbar()
    plt.show()


def main():
    shortedSinusoid = getShortedSinusoid()
    kernel = getKernel()

    shortedSinusoid.plot()

    activation = convolve(shortedSinusoid.signal, kernel.signal, mode='valid')
    activation = Signal(activation, 'Ativação')
    activation.plotAsArray()

    plotSignalAndActivation(shortedSinusoid.signal, activation.signal)

    sinusoid = getSinusoid()
    activation = convolve(sinusoid.signal, kernel.signal, mode='valid')
    activation = Signal(activation, 'Ativação')

    plotSignalAndActivation(sinusoid.signal, activation.signal)


if __name__ == "__main__":
    main()
