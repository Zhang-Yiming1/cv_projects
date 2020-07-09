import numpy as np
import cv2
from matplotlib import pyplot as plt

img2 = cv2.imread('book.jpg')
img1 = cv2.imread('bookIN.jpg')
img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# 寻找关键点并计算描述符
kp1, des1 = sift.detectAndCompute(img1Gray, None)
kp2, des2 = sift.detectAndCompute(img2Gray, None)
# 创建BFMatcher对象
bf = cv2.BFMatcher()
# 蛮力匹配很简单,首先在第一幅图像中选取一个关键点，然后依次与第二幅图像的每一个关键点进行（描述符）距离测试，
# 最后返回距离最近的关键点。
matches = bf.knnMatch(des1, des2, k=2)
# BFMatcher.knnMatch():为每个关键点返回k个最佳匹配（排序之后取前k个），其中k是用户自己设定的

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good[:50], None, flags=2)
# plt.imshow(img3), plt.show()
cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)
cv2.waitKey(50000)
cv2.destroyAllWindows()