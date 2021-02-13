# Data load
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt


class CIFAR10Loader:
    def __init__(self):
        data_transform = transforms.Compose([
                                     transforms.Resize(32),
                                     transforms.ToTensor()])

        self.train_set = datasets.CIFAR10('../../Datasets/dataset',
                                          train=True,
                                          transform=data_transform,
                                          download=True)

        self.test_set = datasets.CIFAR10('../../Datasets/dataset',
                                         train=False,
                                         transform=data_transform,
                                         download=False)

    def getTestLoader(self, batch_size):
        test_loader = DataLoader(self.test_set,
                                 batch_size=batch_size,
                                 shuffle=True)

        return test_loader

    def getTrainLoader(self, batch_size):
        train_loader = DataLoader(self.train_set,
                                  batch_size=batch_size,
                                  shuffle=True)

        return train_loader

    def showTestSamples(self):
        _, axs = plt.subplots(1, 10, figsize=(20, 2))
        for i in range(10):
            data, _ = self.test_set[i]
            axs[i].imshow(data.permute((1, 2, 0)))
            axs[i].axis('off')
        plt.show()
