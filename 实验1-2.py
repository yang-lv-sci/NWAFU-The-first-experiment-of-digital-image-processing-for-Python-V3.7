# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 10:31:27 2023

@author: lvyan
"""

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 读取图像
x = Image.open('SAR-2017-5-1-Export.png').convert('L')  # 转为灰度图像

# 绘制直方图
plt.figure(figsize=(9, 5))
plt.subplot(2, 2, 1)
plt.hist(x.getdata(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

# 显示原始图像
plt.subplot(2, 2, 2)
plt.imshow(x, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示阈值大于200的二值图像
plt.subplot(2, 2, 3)
plt.imshow(x.point(lambda p: p > 200 and 255), cmap='gray')
plt.title('Thresholded Image (>200)')
plt.axis('off')

# 将PIL图像转为NumPy数组，以便使用OpenCV
x_np = np.array(x)

# 使用Otsu's二值化方法
otsu_threshold, w_otsu = cv2.threshold(x_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Otsu's threshold value is: {otsu_threshold}")

# 显示使用Otsu's方法得到的二值图像
plt.subplot(2, 2, 4)
plt.imshow(w_otsu, cmap='gray')
plt.title(f'Binary Image (Otsu\'s method, Threshold={otsu_threshold})')
plt.axis('off')

plt.tight_layout()
plt.savefig("实验1-2直方图阈值二值化.png",dpi=500,bbox_inches="tight")
plt.show()
