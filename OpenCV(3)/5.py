import cv2 
import numpy as np 
from matplotlib import pyplot as plt
img = cv2.imread('6.png')
kernel = np.ones((5,5),np.float32)/25
#cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
#d – Diameter of each pixel neighborhood that is used during filtering. 
# If it is non-positive, it is computed from sigmaSpace 
#9 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差 
dst = cv2.filter2D(img,-1,kernel)
blur = cv2.bilateralFilter(dst,11,75,75)

# BGR2RGB
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(img1),plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(img2),plt.title('Blurred') 
plt.xticks([]), plt.yticks([]) 
plt.show()