#要学习的函数是：cv2.setMouseCallback()
#使用 OpenCV 处理鼠标事件


#以通过执行下列代码查看所有被支持的鼠标事件。

'''
import cv2 
events=[i for i in dir(cv2) if 'EVENT'in i] 
print(events)
'''
'''
'EVENT_FLAG_ALTKEY'
'EVENT_FLAG_CTRLKEY'
'EVENT_FLAG_LBUTTON'
'EVENT_FLAG_MBUTTON'
'EVENT_FLAG_RBUTTON'
'EVENT_FLAG_SHIFTKEY'
'EVENT_LBUTTONDBLCLK'
'EVENT_LBUTTONDOWN'
'EVENT_LBUTTONUP'
'EVENT_MBUTTONDBLCLK'
'EVENT_MBUTTONDOWN'
'EVENT_MBUTTONUP'
'EVENT_MOUSEHWHEEL'
'EVENT_MOUSEMOVE'
'EVENT_MOUSEWHEEL'
'EVENT_RBUTTONDBLCLK'
'EVENT_RBUTTONDOWN'
'EVENT_RBUTTONUP'
'''

#我们的鼠标事件回调函数只用做一件事：在双击过的地方绘制一个圆圈

import cv2 
import numpy as np 
#mouse callback function
def draw_circle(event,x,y,flags,param): 
    if event==cv2.EVENT_LBUTTONDBLCLK: 
        cv2.circle(img,(x,y),20,(255,0,0),-1)

#创建图像与窗口并将窗口与回调函数绑定 
img=np.zeros((512,512,3),np.uint8) 
cv2.namedWindow('image') 
cv2.setMouseCallback('image',draw_circle)
while(1): 
    cv2.imshow('image',img) 
    if cv2.waitKey(20)&0xFF==27: 
        break 
cv2.destroyAllWindows()
