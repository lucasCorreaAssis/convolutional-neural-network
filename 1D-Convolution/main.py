from Signal import Signal
import numpy as np


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


def main():
    # shortedSinusoid = getShortedSinusoid()
    # shortedSinusoid.plotAsArray()

    kernel = getKernel()
    kernel.plotAsArray()
    kernel.invert()
    kernel.plotAsArray()
    kernel.resetSignal()
    kernel.plotAsArray()
    kernel.shift(2)
    kernel.plotAsArray()
    kernel.resetSignal()
    kernel.plotAsArray()


if __name__ == "__main__":
    main()
