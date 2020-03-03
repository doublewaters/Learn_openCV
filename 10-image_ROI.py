import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
img = cv2.imread('0.jpg')
img2 = cv2.imread('0.jpg')


hill = img[100:500,600:900]
img2[100:500,0:300] = hill

plt.subplot(121)
plt.imshow(img)
plt.title("原图")
plt.subplot(122)
plt.imshow(img2)
plt.title('copy后的修改的图')
plt.show()