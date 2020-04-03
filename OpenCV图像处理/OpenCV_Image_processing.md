# OpenCV 中的图像处理 

## 1.颜色空间转换
_目标:_ 
* 你将学习如何对图像进行颜色空间转换，比如从 BGR 到灰度图，或者从 BGR 到 HSV 等。 
* 我没还要创建一个程序用来从一幅图像中获取某个特定颜色的物体。
* 我们将要学习的函数有：cv2.cvtColor()，cv2.inRange() 等。
***
### 1.1 转换颜色空间
在 OpenCV 中有超过 150 中进行颜色空间转换的方法。但是你以后就会 发现我们经常用到的也就两种：BGR↔Gray 和 BGR↔HSV。 
>HSL和HSV都是一种将RGB色彩模型中的点在圆柱坐标系中的表示法。这两种表示法试图做到比基于笛卡尔坐标系的几何结构RGB更加直观。
HSL即色相、饱和度、亮度（英语：Hue, Saturation, Lightness）。HSV即色相、饱和度、明度（英语：Hue, Saturation, Value）
>* 色相（H）是色彩的基本属性，就是平常所说的颜色名称，如红色、黄色等。
>* 饱和度（S）是指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%的数值。
>* 明度（V），亮度（L），取0-100%。
![img]( https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Hsl-hsv_models.svg/800px-Hsl-hsv_models.svg.png 'img')

我们要用到的函数是：cv2.cvtColor(input_image，ﬂag)，其中ﬂag 就是转换类型。
对于BGR↔Gray的转换，我们要使用的ﬂag就是cv2.COLOR_BGR2GRAY。同样对于BGR↔HSV的转换，我们用的ﬂag就是cv2.COLOR_BGR2HSV。你还可以通过下面的命令得到所有可用的ﬂag。
```
import cv2 
flags=[i for in dir(cv2) if i startswith('COLOR_')]
print(flags)
```
以下为输出：
```
['COLOR_BAYER_BG2BGR', 'COLOR_BAYER_BG2BGRA', 'COLOR_BAYER_BG2BGR_EA', 'COLOR_BAYER_BG2BGR_VNG', 'COLOR_BAYER_BG2GRAY', 'COLOR_BAYER_BG2RGB', 'COLOR_BAYER_BG2RGBA', 'COLOR_BAYER_BG2RGB_EA', 'COLOR_BAYER_BG2RGB_VNG', 'COLOR_BAYER_GB2BGR', 'COLOR_BAYER_GB2BGRA', 'COLOR_BAYER_GB2BGR_EA', 'COLOR_BAYER_GB2BGR_VNG', 'COLOR_BAYER_GB2GRAY', 'COLOR_BAYER_GB2RGB', 'COLOR_BAYER_GB2RGBA', 'COLOR_BAYER_GB2RGB_EA', 'COLOR_BAYER_GB2RGB_VNG', 'COLOR_BAYER_GR2BGR', 'COLOR_BAYER_GR2BGRA', 'COLOR_BAYER_GR2BGR_EA', 'COLOR_BAYER_GR2BGR_VNG', 'COLOR_BAYER_GR2GRAY', 'COLOR_BAYER_GR2RGB', 'COLOR_BAYER_GR2RGBA', 'COLOR_BAYER_GR2RGB_EA', 'COLOR_BAYER_GR2RGB_VNG', 'COLOR_BAYER_RG2BGR', 'COLOR_BAYER_RG2BGRA', 'COLOR_BAYER_RG2BGR_EA', 'COLOR_BAYER_RG2BGR_VNG', 'COLOR_BAYER_RG2GRAY', 'COLOR_BAYER_RG2RGB', 'COLOR_BAYER_RG2RGBA', 'COLOR_BAYER_RG2RGB_EA', 'COLOR_BAYER_RG2RGB_VNG', 'COLOR_BGR2BGR555', 'COLOR_BGR2BGR565', 'COLOR_BGR2BGRA', 'COLOR_BGR2GRAY', 'COLOR_BGR2HLS', 'COLOR_BGR2HLS_FULL', 'COLOR_BGR2HSV', 'COLOR_BGR2HSV_FULL', 'COLOR_BGR2LAB', 'COLOR_BGR2LUV', 'COLOR_BGR2Lab', 'COLOR_BGR2Luv', 'COLOR_BGR2RGB', 'COLOR_BGR2RGBA', 'COLOR_BGR2XYZ', 'COLOR_BGR2YCR_CB', 'COLOR_BGR2YCrCb', 'COLOR_BGR2YUV', 'COLOR_BGR2YUV_I420', 'COLOR_BGR2YUV_IYUV', 'COLOR_BGR2YUV_YV12', 'COLOR_BGR5552BGR', 'COLOR_BGR5552BGRA', 'COLOR_BGR5552GRAY', 'COLOR_BGR5552RGB', 'COLOR_BGR5552RGBA', 'COLOR_BGR5652BGR', 'COLOR_BGR5652BGRA', 'COLOR_BGR5652GRAY', 'COLOR_BGR5652RGB', 'COLOR_BGR5652RGBA', 'COLOR_BGRA2BGR', 'COLOR_BGRA2BGR555', 'COLOR_BGRA2BGR565', 'COLOR_BGRA2GRAY', 'COLOR_BGRA2RGB', 'COLOR_BGRA2RGBA', 'COLOR_BGRA2YUV_I420', 'COLOR_BGRA2YUV_IYUV', 'COLOR_BGRA2YUV_YV12', 'COLOR_BayerBG2BGR', 'COLOR_BayerBG2BGRA', 'COLOR_BayerBG2BGR_EA', 'COLOR_BayerBG2BGR_VNG', 'COLOR_BayerBG2GRAY', 'COLOR_BayerBG2RGB', 'COLOR_BayerBG2RGBA', 'COLOR_BayerBG2RGB_EA', 'COLOR_BayerBG2RGB_VNG', 'COLOR_BayerGB2BGR', 'COLOR_BayerGB2BGRA', 'COLOR_BayerGB2BGR_EA', 'COLOR_BayerGB2BGR_VNG', 'COLOR_BayerGB2GRAY', 'COLOR_BayerGB2RGB', 'COLOR_BayerGB2RGBA', 'COLOR_BayerGB2RGB_EA', 'COLOR_BayerGB2RGB_VNG', 'COLOR_BayerGR2BGR', 'COLOR_BayerGR2BGRA', 'COLOR_BayerGR2BGR_EA', 'COLOR_BayerGR2BGR_VNG', 'COLOR_BayerGR2GRAY', 'COLOR_BayerGR2RGB', 'COLOR_BayerGR2RGBA', 'COLOR_BayerGR2RGB_EA', 'COLOR_BayerGR2RGB_VNG', 'COLOR_BayerRG2BGR', 'COLOR_BayerRG2BGRA', 'COLOR_BayerRG2BGR_EA', 'COLOR_BayerRG2BGR_VNG', 'COLOR_BayerRG2GRAY', 'COLOR_BayerRG2RGB', 'COLOR_BayerRG2RGBA', 'COLOR_BayerRG2RGB_EA', 'COLOR_BayerRG2RGB_VNG', 'COLOR_COLORCVT_MAX', 'COLOR_GRAY2BGR', 'COLOR_GRAY2BGR555', 'COLOR_GRAY2BGR565', 'COLOR_GRAY2BGRA', 'COLOR_GRAY2RGB', 'COLOR_GRAY2RGBA', 'COLOR_HLS2BGR', 'COLOR_HLS2BGR_FULL', 'COLOR_HLS2RGB', 'COLOR_HLS2RGB_FULL', 'COLOR_HSV2BGR', 'COLOR_HSV2BGR_FULL', 'COLOR_HSV2RGB', 'COLOR_HSV2RGB_FULL', 'COLOR_LAB2BGR', 'COLOR_LAB2LBGR', 'COLOR_LAB2LRGB', 'COLOR_LAB2RGB', 'COLOR_LBGR2LAB', 'COLOR_LBGR2LUV', 'COLOR_LBGR2Lab', 'COLOR_LBGR2Luv', 'COLOR_LRGB2LAB', 'COLOR_LRGB2LUV', 'COLOR_LRGB2Lab', 'COLOR_LRGB2Luv', 'COLOR_LUV2BGR', 'COLOR_LUV2LBGR', 'COLOR_LUV2LRGB', 'COLOR_LUV2RGB', 'COLOR_Lab2BGR', 'COLOR_Lab2LBGR', 'COLOR_Lab2LRGB', 'COLOR_Lab2RGB', 'COLOR_Luv2BGR', 'COLOR_Luv2LBGR', 'COLOR_Luv2LRGB', 'COLOR_Luv2RGB', 'COLOR_M_RGBA2RGBA', 'COLOR_RGB2BGR', 'COLOR_RGB2BGR555', 'COLOR_RGB2BGR565', 'COLOR_RGB2BGRA', 'COLOR_RGB2GRAY', 'COLOR_RGB2HLS', 'COLOR_RGB2HLS_FULL', 'COLOR_RGB2HSV', 'COLOR_RGB2HSV_FULL', 'COLOR_RGB2LAB', 'COLOR_RGB2LUV', 'COLOR_RGB2Lab', 'COLOR_RGB2Luv', 'COLOR_RGB2RGBA', 'COLOR_RGB2XYZ', 'COLOR_RGB2YCR_CB', 'COLOR_RGB2YCrCb', 'COLOR_RGB2YUV', 'COLOR_RGB2YUV_I420', 'COLOR_RGB2YUV_IYUV', 'COLOR_RGB2YUV_YV12', 'COLOR_RGBA2BGR', 'COLOR_RGBA2BGR555', 'COLOR_RGBA2BGR565', 'COLOR_RGBA2BGRA', 'COLOR_RGBA2GRAY', 'COLOR_RGBA2M_RGBA', 'COLOR_RGBA2RGB', 'COLOR_RGBA2YUV_I420', 'COLOR_RGBA2YUV_IYUV', 'COLOR_RGBA2YUV_YV12', 'COLOR_RGBA2mRGBA', 'COLOR_XYZ2BGR', 'COLOR_XYZ2RGB', 'COLOR_YCR_CB2BGR', 'COLOR_YCR_CB2RGB', 'COLOR_YCrCb2BGR', 'COLOR_YCrCb2RGB', 'COLOR_YUV2BGR', 'COLOR_YUV2BGRA_I420', 'COLOR_YUV2BGRA_IYUV', 'COLOR_YUV2BGRA_NV12', 'COLOR_YUV2BGRA_NV21', 'COLOR_YUV2BGRA_UYNV', 'COLOR_YUV2BGRA_UYVY', 'COLOR_YUV2BGRA_Y422', 'COLOR_YUV2BGRA_YUNV', 'COLOR_YUV2BGRA_YUY2', 'COLOR_YUV2BGRA_YUYV', 'COLOR_YUV2BGRA_YV12', 'COLOR_YUV2BGRA_YVYU', 'COLOR_YUV2BGR_I420', 'COLOR_YUV2BGR_IYUV', 'COLOR_YUV2BGR_NV12', 'COLOR_YUV2BGR_NV21', 'COLOR_YUV2BGR_UYNV', 'COLOR_YUV2BGR_UYVY', 'COLOR_YUV2BGR_Y422', 'COLOR_YUV2BGR_YUNV', 'COLOR_YUV2BGR_YUY2', 'COLOR_YUV2BGR_YUYV', 'COLOR_YUV2BGR_YV12', 'COLOR_YUV2BGR_YVYU', 'COLOR_YUV2GRAY_420', 'COLOR_YUV2GRAY_I420', 'COLOR_YUV2GRAY_IYUV', 'COLOR_YUV2GRAY_NV12', 'COLOR_YUV2GRAY_NV21', 'COLOR_YUV2GRAY_UYNV', 'COLOR_YUV2GRAY_UYVY', 'COLOR_YUV2GRAY_Y422', 'COLOR_YUV2GRAY_YUNV', 'COLOR_YUV2GRAY_YUY2', 'COLOR_YUV2GRAY_YUYV', 'COLOR_YUV2GRAY_YV12', 'COLOR_YUV2GRAY_YVYU', 'COLOR_YUV2RGB', 'COLOR_YUV2RGBA_I420', 'COLOR_YUV2RGBA_IYUV', 'COLOR_YUV2RGBA_NV12', 'COLOR_YUV2RGBA_NV21', 'COLOR_YUV2RGBA_UYNV', 'COLOR_YUV2RGBA_UYVY', 'COLOR_YUV2RGBA_Y422', 'COLOR_YUV2RGBA_YUNV', 'COLOR_YUV2RGBA_YUY2', 'COLOR_YUV2RGBA_YUYV', 'COLOR_YUV2RGBA_YV12', 'COLOR_YUV2RGBA_YVYU', 'COLOR_YUV2RGB_I420', 'COLOR_YUV2RGB_IYUV', 'COLOR_YUV2RGB_NV12', 'COLOR_YUV2RGB_NV21', 'COLOR_YUV2RGB_UYNV', 'COLOR_YUV2RGB_UYVY', 'COLOR_YUV2RGB_Y422', 'COLOR_YUV2RGB_YUNV', 'COLOR_YUV2RGB_YUY2', 'COLOR_YUV2RGB_YUYV', 'COLOR_YUV2RGB_YV12', 'COLOR_YUV2RGB_YVYU', 'COLOR_YUV420P2BGR', 'COLOR_YUV420P2BGRA', 'COLOR_YUV420P2GRAY', 'COLOR_YUV420P2RGB', 'COLOR_YUV420P2RGBA', 'COLOR_YUV420SP2BGR', 'COLOR_YUV420SP2BGRA', 'COLOR_YUV420SP2GRAY', 'COLOR_YUV420SP2RGB', 'COLOR_YUV420SP2RGBA', 'COLOR_YUV420p2BGR', 'COLOR_YUV420p2BGRA', 'COLOR_YUV420p2GRAY', 'COLOR_YUV420p2RGB', 'COLOR_YUV420p2RGBA', 'COLOR_YUV420sp2BGR', 'COLOR_YUV420sp2BGRA', 'COLOR_YUV420sp2GRAY', 'COLOR_YUV420sp2RGB', 'COLOR_YUV420sp2RGBA', 'COLOR_mRGBA2RGBA']
```
>注意:在OpenCV的HSV格式中，H(色彩/色度)的取值范围是[0，179],S(饱和度)的取值范围[0，255],V(亮度)的取值范围[0,255]。但是不同的件使用的值可能不同。所以当你需要拿OpenCV的HSV值与其他软件的HSV值进行对比时，一定要记得归一化。

### 1.2 物体追踪

现在我们知道怎样将一幅图像从 BGR 转换到 HSV 了，我们可以利用这 一点来提取带有某个特定颜色的物体。在 HSV 颜色空间中要比在 BGR 空间 中更容易表示一个特定颜色。在我们的程序中，我们要提取的是一个蓝色的物 体。下面就是就是我们要做的几步： 
* 从视频中获取每一帧图像 
* 将图像转换到 HSV 空间 
* 设置 HSV 阈值到蓝色范围。 
* 获取蓝色物体，当然我们还可以做其他任何我们想做的事，比如：在蓝色 物体周围画一个圈。
  
下面就是我们的代码：
```
import cv2 
import numpy as np
cap=cv2.VideoCapture(0)
while(1):
# 获取每一帧 ret,frame=cap.read()
# 转换到HSV 
hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
# 设定蓝色的阈值 
lower_blue=np.array([110,50,50]) 
upper_blue=np.array([130,255,255])
# 根据阈值构建掩模 
mask=cv2.inRange(hsv,lower_blue,upper_blue)
#对原图像和掩模进行位运算 
res=cv2.bitwise_and(frame,frame,mask=mask)
# 显示图像 
cv2.imshow('frame',frame) 
cv2.imshow('mask',mask) 
cv2.imshow('res',res) 
k=cv2.waitKey(5)&0xFF 
if k==27: 
break
#关闭窗口 
cv2.destroyAllWindows()
```
下图显示了追踪蓝色物体的结果：
![img](2.png '物体追踪')

### 1.3怎样找到要跟踪对象的 HSV 值？

这是我在stackoverﬂow.com上遇到的最普遍的问题。其实这真的很简单， 函数 cv2.cvtColor() 也可以用到这里。但是现在你要传入的参数是（你想要 的）BGR 值而不是一副图。
>**从RGB到HSL或HSV的转换**
>
>设 (r, g, b)分别是一个颜色的红、绿和蓝坐标，它们的值是在0到1之间的实数。设max等价于r, g和b中的最大者。设min等于这些值中的最小者。要找到在HSL空间中的 (h, s, l)值，这里的h ∈ [0, 360）度是角度的色相角，而s, l ∈ [0,1]是饱和度和亮度，计算为：
>$$h=\left\{\begin{array}{ll}
0^{\circ} & \text { if } \max =\min \\
60^{\circ} \times \frac{g-b}{\max -\min }+0^{\circ}, & \text { if } \max =r \text { and } g \geq b \\
60^{\circ} \times \frac{g-b}{\max -\min }+360^{\circ}, & \text { if } \max =r \text { and } g<b \\
60^{\circ} \times \frac{b-r}{\max -\min }+120^{\circ}, & \text { if } \max =g \\
60^{\circ} \times \frac{r-g}{\max -\min }+240^{\circ}, & \text { if } \max =b
\end{array}\right.$$
>$$s=\left\{\begin{array}{ll}
0 & \text { if } l=0 \text { or } m a x=m i n \\
\frac{m a x-m i n}{m a x+m i n}=\frac{m a x-m i n}{2 l}, & \text { if } 0<l \leq \frac{1}{2} \\
\frac{m a x-m i n}{2-(m a x+m i n)}=\frac{m a x-m i n}{2-2 l}, & \text { if } l>\frac{1}{2}
\end{array}\right.$$
>$$l=\frac{1}{2}(\max +\min )$$
>h的值通常规范化到位于0到360°之间。而h = 0用于max = min的（定义为灰色）时候而不是留下h未定义。
HSL和HSV有同样的色相定义，但是其他分量不同。HSV颜色的s和v的值定义如下：
>$$s=\left\{\begin{array}{ll}
0, & \text { if } \max =0 \\
\frac{\max -\min }{\max }=1-\frac{\min }{\max }, & \text { otherwise }
\end{array}\right.$$
>$$\boldsymbol{v}=\max$$


例如，我们要找到绿色的HSV值，我们只需在终端输入以下命令：
```
import cv2
import numpy as np
 
color=np.uint8([[[66 ,99 ,44]]])
hsv_color=cv2.cvtColor(color,cv2.COLOR_BGR2HSV)
 
print(hsv_color)
```


## 2.几何变换

_目标：_
* 学习对图像进行各种几个变换，例如移动，旋转，仿射变换等。 
* 将要学到的函数有：cv2.getPerspectiveTransform。


_变换_
OpenCV提供了两个变换函数，cv2.warpAﬃne和cv2.warpPerspective， 使用这两个函数你可以实现所有类型的变换。cv2.warpAﬃne 接收的参数是 2×3 的变换矩阵，而 cv2.warpPerspective 接收的参数是 3×3 的变换矩 阵。
___
### 2.1 扩展缩放

扩展缩放只是改变图像的尺寸大小。OpenCV 提供的函数 cv2.resize() 可以实现这个功能。图像的尺寸可以自己手动设置，你也可以指定缩放因子。我 们可以选择使用不同的插值方法。在缩放时我们推荐使用cv2.INTER_AREA， 在扩展时我们推荐使用 v2.INTER_CUBIC（慢) 和 v2.INTER_LINEAR。 默认情况下所有改变图像尺寸大小的操作使用的插值方法都是cv2.INTER_LINEAR。 你可以使用下面任意一种方法改变图像的尺寸：

|interpolation选项|所用的插值方法|
|:-------:|:------:|
|INTER_NEAREST|最近邻插值|
|INTER_LINEAR|双线性插值（默认设置）|
|INTER_AREA|使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。|
|INTER_CUBIC|4x4像素邻域的双三次插值|
|INTER_LANCZOS4|8x8像素邻域的Lanczos插值|

代码如下：
```
import cv2 
import numpy as np

img = cv2.imread('0.jpg')
# 方法一：
# 下面的 None 本应该是输出图像的尺寸，但是因为后边我们设置了缩放因子 
# 因此这里为 None 
# res = cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
# OR 
# 这里呢，我们直接设置输出图像的尺寸，所以不用设置缩放因子 
height,width = img.shape[:2] 
res = cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)

while(1): 
    cv2.imshow('res',res) 
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == 27: 
        break 
cv2.destroyAllWindows()
```
### 2.2 平移
平移就是将对象换一个位置。如果你要沿（x，y）方向移动，移动的距离是$\left(t_{x}, t_{y}\right)$，你可以以下面的方式构建移动矩阵： 
$$M=\left[\begin{array}{lll}
1 & 0 & t_{x} \\
0 & 1 & t_{y}
\end{array}\right]$$

你可以使用 Numpy 数组构建这个矩阵（数据类型是 np.ﬂoat32），然 后把它传给函数 cv2.warpAﬃne()。
```
M = np.array([[1, 0,t_x], [0, 1, t_y]], dtype=np.float32)
```


>警告：函数 cv2.warpAﬃne() 的第三个参数的是输出图像的大小，它的格式应该是图像的（宽，高）。应该记住的是图像的宽对应的是列数，高对应的是行数。
```
import numpy as np
import cv2

img = cv2.imread('1.jpg')
cv2.imshow('img',img)
# 平移矩阵[[1, 0, 100], [0, 1, 100]]
M = np.array([[1, 0, 100], [0, 1, 100]], dtype=np.float32)
img_change = cv2.warpAffine(img, M, (1024,768))
cv2.imshow('res', img_change)
cv2.waitKey(0)
```
### 2.3 旋转
对一个图像旋转角度 θ, 需要使用到下面形式的旋转矩阵。
![](https://cdn.mathpix.com/snip/images/BahBR7fIvbiUgoUI1BVmc_KxG5LJPyoQvokGMbpUjcA.original.fullsize.png) 

但是 OpenCV 允许你在任意地方进行旋转，但是旋转矩阵的形式应该修改为
![](https://cdn.mathpix.com/snip/images/v-757zMoK4B5Q9vQC7TcpAb7ay0hAxBM6u5yIh16U30.original.fullsize.png)
其中：
![](https://cdn.mathpix.com/snip/images/tHvxvHqznv4myM7ym5fiStBxz5VfE14_yxBq26-suSY.original.fullsize.png)


为了构建这个旋转矩阵，OpenCV提供了一个函数：cv2.getRotationMatrix2D。 下面的例子是在不缩放的情况下将图像旋转45度。
```
import cv2 
import numpy as np

img=cv2.imread('1.jpg',0)
rows,cols=img.shape
#这里的第一个参数为旋转中心，第二个为旋转角度，第三个为旋转后的缩放因子
#可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)
#第三个参数是输出图像的尺寸中心 
dst=cv2.warpAffine(img,M,(2*cols,2*rows)) 
while(1): 
    cv2.imshow('img',dst) 
    if cv2.waitKey(1)&0xFF==27: 
        break
cv2.destroyAllWindows()
```
### 2.4 仿射变换

在仿射变换中，原图中所有的平行线在结果图像中同样平行。为了创建这个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。然后cv2.getAﬃneTransform 会创建一个2x3的矩阵，最后这个矩阵会被传给函数cv2.warpAﬃne。

标记的点为蓝色

```
import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img=cv2.imread('4.png') 
rows,cols,ch=img.shape
pts1=np.float32([[100,100],[200,100],[50,450]])
pts2=np.float32([[50,50],[150,50],[100,500]])
M=cv2.getAffineTransform(pts1,pts2)
dst=cv2.warpAffine(img,M,(cols,rows))
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```

结果如下图：
![img](5.png '仿射变换')

### 2.5透视变换 

对于视角变换，我们需要一个 3x3 变换矩阵。在变换前后直线还是直线。 要构建这个变换矩阵，你需要在输入图像上找 4 个点，以及他们在输出图 像上对应的位置。这四个点中的任意三个都不能共线。这个变换矩阵可以有 函数 cv2.getPerspectiveTransform() 构建。然后把这个矩阵传给函数 cv2.warpPerspective。 

```
import cv2 
import numpy as np 
from matplotlib import pyplot as plt

img=cv2.imread('6.png') 
rows,cols,ch=img.shape

pts1 = np.float32([[15,38],[342,44],[40,394],[374,361]]) 
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
```
结果如下图：
![img](7.png '透视变换')

