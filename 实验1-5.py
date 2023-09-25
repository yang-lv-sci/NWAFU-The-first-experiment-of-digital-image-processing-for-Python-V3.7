# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:57:19 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
x = cv2.imread('SAR-2017-5-1-Export.png', cv2.IMREAD_GRAYSCALE)

# Create a 2x2 grid for plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 7))

# Display original image on the top-left
axs[0, 0].imshow(x, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')  # Turn off axis numbers and ticks

sizes = [7, 9, 11]
titles = ['7x7 Gaussian Filter', '9x9 Gaussian Filter', '11x11 Gaussian Filter']

# Apply each filter size and display the results
for i, size in enumerate(sizes):
    h = cv2.getGaussianKernel(size, -1)  # -1 means sigma is calculated from size
    h = h @ h.T
    y = cv2.filter2D(x, -1, h)

    row = (i + 1) // 2
    col = (i + 1) % 2
    axs[row, col].imshow(y, cmap='gray')
    axs[row, col].set_title(titles[i])
    axs[row, col].axis('off')  # Turn off axis numbers and ticks

plt.tight_layout()  # Adjust spacing between subplots
plt.savefig("实验1-5-1高斯模板滤波.png",dpi=500,bbox_inches="tight")
plt.show()


# Equivalent to MATLAB's fspecial('gaussian', size, sigma)
def fspecial_gaussian(size, sigma):
    center = (size - 1) / 2
    x = np.arange(0, size) - center
    y = x[:, np.newaxis]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    return g / g.sum()

# Define the Gaussian filters
h1 = fspecial_gaussian(100, 3)
h2 = fspecial_gaussian(100, 10)

# Equivalent to MATLAB's meshgrid function
x, y = np.meshgrid(np.arange(1, 101), np.arange(1, 101))

# Create mesh plots
fig = plt.figure(figsize=(12, 6))

# Create a 3D axis for h1
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(x, y, h1, cmap='viridis')
ax1.set_title("Gaussian filter with sigma=3")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")

# Create a 3D axis for h2
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(x, y, h2, cmap='viridis')
ax2.set_title("Gaussian filter with sigma=10")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.savefig("实验1-5-2高斯模板图像.png",dpi=500,bbox_inches="tight")
plt.show()