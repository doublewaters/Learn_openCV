import cv2 
import numpy as np 
from matplotlib import pyplot as plt
img = cv2.imread('opencv-n.png')
median = cv2.medianBlur(img,9)


# BGR2RGB
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(median, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(img1),plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(img2),plt.title('Blurred') 
plt.xticks([]), plt.yticks([]) 
plt.show()