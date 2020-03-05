import numpy as np
import cv2

img = cv2.imread('0.jpg')
cv2.imshow('img', img)
# 平移矩阵[[1,0,-100],[0,1,-12]]
M = np.array([[1, 0, -100], [0, 1, -12]], dtype=np.float32)
img_change = cv2.warpAffine(img, M, (300, 300))
cv2.imshow('res', img_change)
cv2.waitKey(0)
