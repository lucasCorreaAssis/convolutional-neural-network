# 1D Convolution
Before we get into the image domain, we will understand convolution in a simpler context, working with one dimension signals.

**Note:** Convolution is the summation of the product between functions, which one of them is inverted and shifted.

We'll assume the following problem:

You decided to collect data from a phone's accelerometer. The objective consists that people carry their phones in their pockets to further analyse how the sensor reacts to this situation. The signal could be represented as a noised sinusoid.

## Kernel
On the context of image processing, kernel is a convolutional filter. Basically it's a n-dimensions matrix which will be operated with the data through a convolution.

Therefore we need to propose a kerenel which simulates the pattern we're looking for: increasing intervals

**Note:** Convolution operates the functions after inverting the kernel