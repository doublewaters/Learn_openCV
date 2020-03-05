#cv2.line()，cv2.circle()，cv2.rectangle()， cv2.ellipse()，cv2.putText() 

# 绘图函数需要设置下面这些参数： 
# • img：你想要绘制图形的那幅图像。 
# • color：形状的颜色。以RGB为例，需要传入一个元组，例如：255,0,0）代表蓝色。对于灰度图只需要传入灰度值。 
# • thickness：线条的粗细。如果给一个闭合图形设置为 -1，那么这个图形 就会被填充。默认值是 1.
# • linetype：线条的类型，8连接，抗锯齿等。默认情况是8连接。cv2.LINE_AA 为抗锯齿，这样看起来会非常平滑。

import numpy as np
import cv2 

# Create a black image
img = np.zeros((512,512,3), np.uint8)

#Draw a diagonal blue line with thickness of 5 px
#只需要告诉函数这条线的起点和终点。
cv2.line(img,(0,0),(511,511),(255,0,0),5) 

#矩形：你需要告诉函数的左上角顶点和右下角顶点的坐标。
cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

#圆：只需要指定圆形的中心点坐标和半径大小。
cv2.circle(img,(447,63), 63, (0,0,255), -1)

#椭圆:一个参数是中心点的位置坐标。 下一个参数是长轴和短轴的长度。椭圆沿逆时针方向旋转的角度。椭圆弧演 顺时针方向起始的角度和结束角度，如果是 0 很 360，就是整个椭圆
cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

#多边形：需要指点每个顶点的坐标。
pts=np.array([[30,5],[110,80],[100,120],[10,10]], np.int32) 
pts=pts.reshape((-1,1,2)) 
cv2.polylines(img,[pts], 1, (255,0,255),2)
# 这里 reshape 的第一个参数为 -1, 表明这一维的长度是根据后面的维度的计算出来的。
# 注意:如果第三个参数是False,我们得到的多边形是不闭合的(首尾不相连)。

#文字
# 要在图片上绘制文字，你需要设置下列参数：
#     • 你要绘制的文字
#     • 你要绘制的位置
#     • 字体类型（通过查看 cv2.putText() 的文档找到支持的字体）
#     • 字体的大小 
#     • 文字的一般属性如颜色，粗细，线条的类型等。为了更好看一点推荐使用 linetype=cv2.LINE_AA。 
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2)



winname = 'example'
cv2.namedWindow(winname)
cv2.imshow(winname, img) 
cv2.waitKey(0)
cv2.destroyWindow(winname)

