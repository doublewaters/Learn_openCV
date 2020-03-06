import numpy as np
import cv2
from matplotlib import pyplot as plt 

img = cv2.imread('0.jpg',1)

# plt.imshow(img2, cmap = 'gray', interpolation = 'bicubic')
# plt.xticks([]), plt.yticks([])      # to hide tick values on X and Y axis 
# plt.show()


# 注意：彩色图像使用 OpenCV 加载时是 BGR 模式。但是 Matplotib 是 RGB模式。所以彩色图像如果已经被 OpenCV 读取，那它将不会被 Matplotib 正确显示。

#方法1：
#利用cv2.split()和cv2.merge()函数将加载的图像先按BGR模式分割开来,然后再以RGB模式合并图像
b, g, r = cv2.split(img)
img2 = cv2.merge([r, g, b])

#方法2：
#数组逆序 将原本的BGR转化为RGB,省去了分割色彩
img3 = img[..., ::-1]  

#方法3：
#使用OpenCV自带的模式转换函数cv2.cvtColor()
img4 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.subplot(221)
plt.imshow(img)
plt.title("BGR")
plt.subplot(222)
plt.imshow(img2)
plt.title('RGB')
plt.subplot(223)
plt.imshow(img3)
plt.title('RGB')
plt.subplot(224)
plt.imshow(img4)
plt.title('RGB')
plt.show()