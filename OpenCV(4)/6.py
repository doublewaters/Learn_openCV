
import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('j.png',0)
kernel = np.ones((9,9),np.uint8) 
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(tophat, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(img1),plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(img2),plt.title('Tophat') 
plt.xticks([]), plt.yticks([]) 
plt.show()
