import torch
from sklearn.metrics import accuracy_score
import numpy as np
import time


def train(train_loader, net, epoch, criterion, optimizer, **kwargs):
    # Training mode
    net.train()

    start = time.time()

    epoch_loss = []
    pred_list, target_list = [], []
    for batch in train_loader:

        data, target = batch

        # Cast data in GPU
        data = data.to(kwargs['device'])
        target = target.to(kwargs['device'])

        # Forward
        ypred = net(data)
        loss = criterion(ypred, target)
        epoch_loss.append(loss.cpu().data)

        _, pred = torch.max(ypred, axis=1)
        pred_list.append(pred.cpu().numpy())
        target_list.append(target.cpu().numpy())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = np.asarray(epoch_loss)
    pred_list = np.asarray(pred_list).ravel()
    target_list = np.asarray(target_list).ravel()

    acc = accuracy_score(pred_list, target_list)

    end = time.time()
    print('#################### Train ####################')
    print('Epoch %d, Loss: %.4f +/- %.4f, Acc: %.2f, Time: %.2f' % (epoch,
                                                                    epoch_loss.mean(),
                                                                    epoch_loss.std(),
                                                                    acc*100,
                                                                    end-start))

    return epoch_loss.mean()
