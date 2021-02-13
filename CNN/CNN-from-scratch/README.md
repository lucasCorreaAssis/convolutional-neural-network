# CNN from scratch

Here we will implement the most traditional train strategy, the train from scratch, applying it for image classifier. My strategy was to implement the LeNet, the first successfully CNN implemented.


    |----------------------------------------------------------------------|
    |                     | Feature |       | Kernel |        |            |
    |       Layer         |   map   | Size  |  size  | Stride | Activation |
    |----------------------------------------------------------------------|
    | Input |    Image    |    1    | 32x32 |    -   |    -   |     -      |
    |----------------------------------------------------------------------|
    |   1   | Convolution |    1    | 28x28 |   5x5  |    1   |    tanh    |
    |----------------------------------------------------------------------|
    |       |   Average   |         |       |        |        |            |
    |   2   |   Pooling   |    6    | 14x14 |   2x2  |    2   |    tanh    |
    |----------------------------------------------------------------------|
    |   3   | Convolution |    16   | 10x10 |   5x5  |    1   |    tanh    |
    |----------------------------------------------------------------------|
    |       |   Average   |         |       |        |        |            |
    |   4   |   Pooling   |    16   |  5x5  |   2x2  |    2   |    tanh    |
    |----------------------------------------------------------------------|
    |   5   | Convolution |    120  |  1x1  |   5x5  |    1   |    tanh    |
    |----------------------------------------------------------------------|
    |   6   |      FC     |    -    |   84  |    -   |    -   |    tanh    |
    |----------------------------------------------------------------------|
    | Output|      FC     |    -    |   10  |    -   |    -   |   softmax  |
    |----------------------------------------------------------------------|


## Training
The first step is to define the algorithms which will be used in the training process:
* Loss function, which will evaluate the performance of each training step
* Optimizer, which will define the best way to update the weights from the loss function.

### Training Flow
* iterate through epochs
* iterate through batches
* Cast the data in the hardware
* Forward and loss calculation
* Reset the gradient from the optimizer
* Gradient calculation and weights update

To follow the model's convergence (and ensure everything worked fine), at the end of each epoch we can print the losses' average and 
standard deviation of each iteration

## Validation
At this stage, pyTorch offers two options:
* ```model.eval()```: impacts the net's forward, informing the layers in case its behavior changes between flows (ex. dropout).
* ```with torch.no_grad()```: Context manager which deactivate the calculation and storage of gradients (time and storage saving). All validation code must be executed in this context.

Example:

```python
net.eval()
with torch.no_grad():
    for batch in teste_loader:
        # Código de validação```

