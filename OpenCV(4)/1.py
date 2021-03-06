import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('j.png',0) 
kernel = np.ones((5,5),np.uint8) 
erosion = cv2.erode(img,kernel,iterations = 1)

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(img1),plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(img2),plt.title('Erosion') 
plt.xticks([]), plt.yticks([]) 
plt.show()
