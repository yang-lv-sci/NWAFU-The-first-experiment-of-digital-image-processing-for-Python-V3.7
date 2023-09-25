# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:34:34 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
x = cv2.imread('SAR-2017-5-1-Export.png', cv2.IMREAD_GRAYSCALE)  # Assuming the image is grayscale as in the example

# Define the sharpening kernel
h = np.array([[-1, -1, -1],
              [-1,  8, -1],
              [-1, -1, -1]])

# Apply the filter
y = cv2.filter2D(x, -1, h)

# Display the original and sharpened images side by side using matplotlib
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(x, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(y, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.tight_layout()
plt.savefig("实验1-7图像锐化.png",dpi=500,bbox_inches="tight")
plt.show()
