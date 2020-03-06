#获取图像属性

import cv2 
import numpy as np 
img=cv2.imread('./0.jpg') 
#img.shape 可以获取图像的形状。他的返回值是一个包含行数，列数， 通道数的元组。
print(img.shape)
#注意:如果图像是灰度图，返回值仅有行数和列数。所以通过检查这个返回值就可以知道加载的是灰度图还是彩色图。

#img.size 可以返回图像的像素数目
print(img.size)

#img.dtype 返回的是图像的数据类型.
print(img.dtype)

#注意：在debug时 img.dtype 非常重要。因为在 OpenCVPython 代码中经常出现数据类型的不一致。
