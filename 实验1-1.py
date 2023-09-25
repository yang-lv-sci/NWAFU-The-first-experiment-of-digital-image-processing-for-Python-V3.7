# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 09:56:11 2023

@author: lvyan
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio  # <-- Use version 2 of imageio
from skimage import exposure

# Load the image
x = imageio.imread('SAR-2017-5-1-Export.png')
# Set figure size

plt.figure(figsize=(9, 5))  # <-- Setting figure size

# Plot original image and its histogram
plt.subplot(4, 2, 1)
plt.imshow(x, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.hist(x.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Original Image')

# Adjust the image
y = exposure.rescale_intensity(x, in_range='image', out_range=(0.1 * 255, 0.5 * 255))
y = np.asarray(y, dtype=np.uint8)

# Plot adjusted image and its histogram
plt.subplot(4, 2, 3)
plt.imshow(y, cmap='gray')
plt.title('Adjusted Image-1-5')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.hist(y.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Adjusted Image')

# Adjust the image
y = exposure.rescale_intensity(x, in_range='image', out_range=(0.4 * 255, 0.6 * 255))
y = np.asarray(y, dtype=np.uint8)

# Plot adjusted image and its histogram
plt.subplot(4, 2, 5)
plt.imshow(y, cmap='gray')
plt.title('Adjusted Image-4-6')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.hist(y.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Adjusted Image')

# Adjust the image
y = exposure.rescale_intensity(x, in_range='image', out_range=(0.5 * 255, 1.0 * 255))
y = np.asarray(y, dtype=np.uint8)

# Plot adjusted image and its histogram
plt.subplot(4, 2, 7)
plt.imshow(y, cmap='gray')
plt.title('Adjusted Image-5-10')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.hist(y.ravel(), bins=256, histtype='step', color='black')
plt.title('Histogram of Adjusted Image')

plt.tight_layout()
plt.savefig("实验1-1直方图.png",dpi=500,bbox_inches="tight")
plt.show()
