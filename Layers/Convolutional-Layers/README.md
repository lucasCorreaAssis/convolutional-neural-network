## Convolutional Layers

conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)

* For the first layer:
    * Input depth (number of channels [in_channels])
    * Outpu depth (number of activation maps [out_channels])

* View field: kernel_size
* Stride: step size
* Padding: Filling the data with zeros (artificial borders)
* Activation map's spatial resolution:
    * out = ( ( in - view_field + 2 * padding ) / stride ) + 1