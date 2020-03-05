import cv2 
import numpy as np
from matplotlib import pyplot as plt
#加载图像 
img0_1 = cv2.imread('opencv.png')
img0_2 = cv2.imread('1.png') 
img1 = cv2.imread('opencv.png')
img2 = cv2.imread('1.png') 
# I want to put logo on top-left corner, So I create a ROI 
rows,cols,channels = img2.shape 
roi = img1[0:rows, 0:cols ]
# Now create a mask of logo and create its inverse mask also 
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY) 
ret,mask = cv2.threshold(img2gray, 175, 255, cv2.THRESH_BINARY) 
mask_inv = cv2.bitwise_not(mask)
# Now black-out the area of logo in ROI 
#取roi中与mask 中不为零的值对应的像素的值，其他值为0
#注意这里必须有mask=mask或者mask=mask_inv,其中的mask=不能忽略
img1_bg = cv2.bitwise_and(roi,roi,mask = mask) 
#取roi中与mask_inv中不为零的值对应的像素的值，其他值为0。
# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
# Put logo in ROI and modify the main image 
dst = cv2.add(img1_bg,img2_fg) 
img1[0:rows, 0:cols ] = dst
# cv2.imshow('res',img1) 
# cv2.imwrite('13_3.png',img1)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()
def show(n,img,tittle):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2,5,n)
    plt.imshow(img)
    plt.title(tittle)
show(1,img0_1,'img0_1')
show(2,img0_2,'img0_2')
show(3,img2gray,'img2gray')
show(4,mask,'mask')
show(5,mask_inv,'mask_inv')
show(6,img1_bg,'img1_bg')
show(7,img2_fg,'img2_fg')
show(8,roi,'roi')
show(9,dst,'dst')
show(10,img1,'img1')

plt.show()