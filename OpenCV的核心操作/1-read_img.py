import numpy as np
import cv2

#load a color image in grayscale
#img = cv2.imread('0.jpg',0)         #灰度
img = cv2.imread('0.jpg',1)         #彩色
# #提示：路径错误，不提示

# #show a image 
# cv2.namedWindow('image', cv2.WINDOW_NORMAL) 
cv2.imshow('image',img)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# #save a image
# cv2.imwrite('111.png',img)

#下面的程序将会加载一个灰度图，显示图片，按下’s’键保存后退出，或者 按下 ESC 键退出不保存。
k = cv2.waitKey(0)&0xff
if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('1.png',img)
    cv2.destroyAllWindows()