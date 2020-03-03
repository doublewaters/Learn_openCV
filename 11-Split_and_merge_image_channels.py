#Split and merge image channels

import cv2 
import numpy as np
from matplotlib import pyplot as plt 

img=cv2.imread('0.jpg',cv2.IMREAD_UNCHANGED) 
#把 BGR 拆 分成单个通道
b,g,r=cv2.split(img) 
#警告：cv2.split() 是一个比较耗时的操作。只有真正需要时才用它，能用 Numpy 索引就尽量用。

'''
#或者可以直接使用 Numpy 索引，这会更快
b=img[:,:,0]
'''

img = img[..., ::-1]

#将bgr三通道合并
#img=cv2.merge(b,g,r)

plt.subplot(221)
plt.imshow(img)
plt.title("BGR")
plt.subplot(222)
plt.imshow(b)
plt.title('B')
plt.subplot(223)
plt.imshow(g)
plt.title('G')
plt.subplot(224)
plt.imshow(r)
plt.title('R')
plt.show()

