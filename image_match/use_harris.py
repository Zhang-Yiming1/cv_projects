import cv2
import numpy as np

# detector parameters
block_size = 3
sobel_size = 3
k = 0.04

img = cv2.imread('./电力塔.jpg')
cv2.imshow('origin', img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# modify the data type setting to 32-bit floating point
gray_img = np.float32(gray_img)

# detect the corners with appropriate values as input parameters
dst = cv2.cornerHarris(gray_img, block_size, sobel_size, k)
# dst = cv2.dilate(dst, None)
img[dst > 0.01*dst.max()] = [0, 0, 255]

cv2.imshow('corners_img', img)
cv2.waitKey(50000)
cv2.destroyAllWindows()