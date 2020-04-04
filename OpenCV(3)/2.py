import cv2 
import numpy as np 
from matplotlib import pyplot as plt
img = cv2.imread('opencv.png')
blur = cv2.blur(img,(5,5))

# BGR2RGB
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

plt.subplot(121),plt.imshow(img1),plt.title('Original') 
plt.xticks([]), plt.yticks([]) 
plt.subplot(122),plt.imshow(img2),plt.title('Blurred') 
plt.xticks([]), plt.yticks([]) 
plt.show()