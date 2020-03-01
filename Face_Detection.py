import cv2
import numpy as ny 

#读入一幅图像   显示一幅图像    保存一幅图像


face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml') 
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')

cam = cv2.VideoCapture(0)
img = cam.read()
#img = cv2.imread('0.jpg',0)
#gray = cv2.cvtColor(img,)
gray = img

faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
for (x,y,w,h) in faces: 
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1) 
    roi_gray = gray[y:y+h, x:x+w] 
    roi_color = img[y:y+h, x:x+w] 
    eyes = eye_cascade.detectMultiScale(roi_gray) 
    for (ex,ey,ew,eh) in eyes: 
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

cv2.imshow("0",img)
cv2.waitKey(500000)