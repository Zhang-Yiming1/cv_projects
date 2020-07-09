
import cv2
import numpy as np

image_fold = './caixukun.png'
img = cv2.imread(image_fold)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建sift特征提取器
sift = cv2.xfeatures2d.SIFT_create()
'''
创建一个SIFT对象:
cv2.xfeatures2d.SIFT_create(, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)

nfeatures：默认为0，要保留的最佳特征的数量。 特征按其分数排名（在SIFT算法中按局部对比度排序）
nOctaveLayers：默认为3，金字塔每组(Octave)有多少层。 3是D. Lowe纸中使用的值。
contrastThreshold：默认为0.04，对比度阈值，用于滤除低对比度区域中的弱特征。 阈值越大，检测器产生的特征越少。
edgeThreshold：默认为10，用来过滤边缘特征的阈值。注意，它的意思与contrastThreshold不同，edgeThreshold越大，滤出的特征越少（保留更多特征）。
sigma：默认为1.6，高斯金字塔中的σ。
'''
# keypoints = sift.detect(imgGray, None)
# 可以调用sift.compute（）来计算找到的关键点的描述符, 例如：kp，des = sift.compute（gary，kp）
'''
检测特征点:
sift.detect(image,keypoints)  或  keypoint = sift.detect(image, None)

sift：配置好SIFT算法对象
image：输入图像，单通道
keypoint：输出参数，保存着特征点，每个特征点包含有以下信息：
                         Point2f pt：坐标 
                         float size：特征点的邻域直径 
                         float angle：特征点的方向，值为[0,360度)，负值表示不使用 
                         float response; 
                         int octave：特征点所在的图像金字塔的组 
                         int class_id：用于聚类的id 
'''
# 计算sift特征点的关键点和相应的描述子。
keypoints, descriptor = sift.detectAndCompute(imgGray, None)


outImage = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.namedWindow('outImage', cv2.WINDOW_NORMAL)
cv2.imshow('outImage', outImage)
cv2.waitKey(50000)

'''
绘制特征点：
cv2.drawKeypoint(image, keypoints, outImage, color, flags)

或：outImage = cv2.drawKeypoint(image, keypoints, None, color, flags)

image：输入图像
keypoints：上面获取的特征点
outImage：输出图像
color：颜色，默认为随机颜色
flags：绘制点的模式，有以下四种模式
cv2.DRAW_MATCHES_FLAGS_DEFAULT：

默认值，只绘制特征点的坐标点,显示在图像上就是一个个小圆点,每个小圆点的圆心坐标都是特征点的坐标。

cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS：

绘制特征点的时候绘制的是带有方向的圆,这种方法同时显示图像的坐标,size，和方向,是最能显示特征的一种绘制方式。

cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG：

只绘制特征点的坐标点,显示在图像上就是一个个小圆点,每个小圆点的圆心坐标都是特征点的坐标。

cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINT：

 单点的特征点不被绘制 

'''
