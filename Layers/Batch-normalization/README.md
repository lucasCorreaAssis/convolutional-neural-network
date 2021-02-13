# Normalization

Pre-processing stage of the data, which has the purpose of reducing computational power required to process the data.

The authors of **batch normalization** stated that this difference in the distributions could affect the network's convergence. To solve this problem, they proposed an additional layer to be placed after the convolution and before the non-linear activation.

## Implementation

conv = torch.nn.BatchNorm2d(num_features)

* num_features: Number of channels (activation map's) from the previous layer

My strategy was to implement a convolutional block:
* Convolutional layer
* Batch normalization
* Non-linear activation
* Pooling