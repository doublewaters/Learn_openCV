
import cv2
import numpy as np

def BGR2HSV(B,G,R):
    color=np.uint8([[[B ,G ,R]]])
    hsv_color=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
    return hsv_color

if __name__ == "__main__":
    input(b1, g1, r1)
    hsv = BGR2HSV(b1,g1,r1)
    print(hsv)

    




# color=np.uint8([[[66 ,99 ,44]]])
# hsv_color=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
 
# print(hsv_color)