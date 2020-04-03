import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img=cv2.imread('6.png') 
rows,cols,ch=img.shape

pts1 = np.float32([[15,38],[342,44],[40,394],[374,361]]) 
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()