import cv2 
import numpy as np 

#读入一张图片
img = cv2.imread('0.jpg') 

px = img[100,100]
#返回为BGR的数组

blue = img[100,100,0]   #第三的值为0，1，2，代表B、G、R
green = px[1]

print(px)
print(blue)
print(green)

print(img.item(100,100,2))      #输出R值
img.itemset((100,100,2),100)    #修改R值
print(img.item(100,100,2))          

img[100,100]=[255,255,255]      #修改像素值为[255,255,255]
print(img[100,100])
