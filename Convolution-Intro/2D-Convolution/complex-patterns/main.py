'''We'll use a patch from the image to use as
   convolutional filter. Using the image plane.jpg,
   we selected the turbine region at [109, 129, 255, 275]'''

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from Filter.Filter import Filter

# Read the image and select the rectangle
img = io.imread('samples/plane.jpg')
img = img[:, :, 0]
rectangle = [109, 129, 255, 275]

# Print the image and rectangle
fig, ax = plt.subplots()
ax.imshow(img, cmap='gray')
ax.add_patch(patches.Rectangle((rectangle[2], rectangle[0]),
                               (rectangle[3] - rectangle[2]),
                               (rectangle[1] - rectangle[0]),
                               color='red',
                               fill=False))

plt.show()


# Adjusting the patch:
#  * subtract the patch by the average pixel value
#  * Flip the filter
patch = img[rectangle[0]:rectangle[1], rectangle[2]:rectangle[3]]
patch = patch - patch.mean()
patch = np.flip(patch)

plt.imshow(patch, cmap='gray')
plt.show()

# Applying the filter to the image
complex_filter = Filter(img, patch, 'Complex Patterns')
complex_filter.convolveFilter()
complex_filter.plotResult()
