# coding=utf-8
import cv2
import numpy as np

img = cv2.imread("./lena.png", 0)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

absX = cv2.convertScaleAbs(x) # 转回uint8
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# (3, 3)表示高斯矩阵的长与宽都是3，标准差取0
img_blur = cv2.GaussianBlur(img, (3, 3), 0)
canny = cv2.Canny(img_blur, 50, 150)

# cv2.imshow("absX", absX)
# cv2.imshow("absY", absY)
cv2.imshow("Result", dst)
cv2.imshow('blur_img', img_blur)
cv2.imshow('Canny', canny)

cv2.waitKey(50000)
cv2.destroyAllWindows()