# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 12:12:04 2023

@author: lvyan
"""
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.filters import sobel,prewitt,laplace
from skimage.feature import canny
import cv2
from matplotlib.gridspec import GridSpec
# 读取图像
x = cv2.imread('SAR-2017-5-1-Export.png', cv2.IMREAD_GRAYSCALE)

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

ax2.imshow(sobel(x), cmap='gray')
ax2.set_title("sobel Image")
ax2.axis('off')

ax3.imshow(canny(x), cmap='gray')
ax3.set_title("canny Image")
ax3.axis('off')

ax4.imshow(prewitt(x), cmap='gray')
ax4.set_title("prewitt Image")
ax4.axis('off')

ax5.imshow(laplace(x), cmap='gray')
ax5.set_title("laplace Image")
ax5.axis('off')

plt.tight_layout()
plt.savefig("实验1-8不同算子滤波方法比较.png",dpi=500,bbox_inches="tight")
plt.show()