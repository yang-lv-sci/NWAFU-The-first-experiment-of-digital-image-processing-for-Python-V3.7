# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:11:58 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 读取图像
x = cv2.imread('SAR-2017-5-1-Export.png', cv2.IMREAD_GRAYSCALE)

# 添加胡椒盐噪声
def salt_and_pepper_noise(image, amount=0.22):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, int(num_salt)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255

    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

h = salt_and_pepper_noise(x, 0.22)

# 5x5 均值滤波
mean_filtered = cv2.blur(h, (5,5))

# 5x5 高斯滤波
gaussian_filtered = cv2.GaussianBlur(h, (5,5), 0)

# 中值滤波
median_filtered = cv2.medianBlur(h, 5)

# 使用 Matplotlib 显示图像
fig = plt.figure(figsize=(10, 5))

gs = GridSpec(2, 6, figure=fig) # 2行，6列的网格
ax1 = fig.add_subplot(gs[0, 1:3])  # Original Image
ax2 = fig.add_subplot(gs[0, 3:5])  # Noisy Image
ax3 = fig.add_subplot(gs[1, 0:2])  # Mean Filtered Image
ax4 = fig.add_subplot(gs[1, 2:4])  # Gaussian Filtered Image
ax5 = fig.add_subplot(gs[1, 4:6])  # Median Filtered Image

ax1.imshow(x, cmap='gray')
ax1.set_title("Original Image")
ax1.axis('off')

ax2.imshow(h, cmap='gray')
ax2.set_title("Noisy Image")
ax2.axis('off')

ax3.imshow(mean_filtered, cmap='gray')
ax3.set_title("Mean Filtered Image")
ax3.axis('off')

ax4.imshow(gaussian_filtered, cmap='gray')
ax4.set_title("Gaussian Filtered Image")
ax4.axis('off')

ax5.imshow(median_filtered, cmap='gray')
ax5.set_title("Median Filtered Image")
ax5.axis('off')

plt.tight_layout()
plt.savefig("实验1-6椒盐噪声滤波方法比较.png",dpi=500,bbox_inches="tight")
plt.show()
