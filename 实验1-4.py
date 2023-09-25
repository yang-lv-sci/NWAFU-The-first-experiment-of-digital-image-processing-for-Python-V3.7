# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:49:14 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_average_filter(img, ksize):
    h = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
    return cv2.filter2D(img, -1, h)

x = cv2.imread('SAR-2017-5-1-Export.png', cv2.IMREAD_GRAYSCALE)
# Assuming the image is grayscale
y_3x3 = apply_average_filter(x, 3)
y_5x5 = apply_average_filter(x, 5)
y_7x7 = apply_average_filter(x, 7)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Displaying images
axs[0, 0].imshow(x, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 1].imshow(y_3x3, cmap='gray')
axs[0, 1].set_title('3x3 Average Filter')
axs[1, 0].imshow(y_5x5, cmap='gray')
axs[1, 0].set_title('5x5 Average Filter')
axs[1, 1].imshow(y_7x7, cmap='gray')
axs[1, 1].set_title('7x7 Average Filter')

# Removing axis ticks
for ax in axs.ravel():
    ax.axis('off')

plt.tight_layout()
plt.savefig("实验1-4领域平均模板滤波.png",dpi=500,bbox_inches="tight")
plt.show()
