# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:42:00 2023

@author: lvyan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读入图像
x = cv2.imread('SAR-2017-5-1-Export.png', cv2.IMREAD_GRAYSCALE)

# 显示原始图像
plt.figure(figsize=(9,7))
plt.subplot(2, 2, 1)
plt.imshow(x, cmap='gray')
plt.title("Original Image")

# 显示原始图像的直方图
hist, bins = np.histogram(x.flatten(),256,[0,256])
plt.subplot(2, 2, 2)
plt.plot(hist)
plt.title("Original Histogram")

# 计算累计直方图
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.subplot(2, 2, 3)
plt.plot(cdf_normalized, color = 'b')
plt.title("Cumulative Histogram")

# 直方图均衡化
y = cv2.equalizeHist(x)

plt.subplot(2, 2, 4)
plt.imshow(y, cmap='gray')
plt.title("Equalized Image")

plt.savefig("实验1-3直方图均衡化.png",dpi=500,bbox_inches="tight")
plt.show()
