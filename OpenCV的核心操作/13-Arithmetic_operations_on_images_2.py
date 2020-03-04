
import cv2 
import numpy as np

img1=cv2.imread('1.png') 
img2=cv2.imread('opencv.png')

dst=cv2.addWeighted(img1,0.7,img2,0.3,0)
cv2.imshow('dst',dst) 
cv2.waitKey(0) 
cv2.imwrite('13.png',dst)
cv2.destroyAllWindow()
