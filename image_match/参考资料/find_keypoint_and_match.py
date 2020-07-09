import cv2
import numpy as np


class FindKeyPointsAndMatching:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute = cv2.BFMatcher()

    def get_key_points(self, img1, img2):
        g_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        kp1, kp2 = {}, {}
        print('=======>Detecting key points!')
        kp1['kp'], kp1['des'] = self.sift.detectAndCompute(g_img1, None)
        kp2['kp'], kp2['des'] = self.sift.detectAndCompute(g_img2, None)
        return kp1, kp2

    def match(self, kp1, kp2):
        print('=======>Matching key points!')
        matches = self.brute.knnMatch(kp1['des'], kp2['des'], k=2)
        # BFMatcher.knnMatch():为每个关键点返回k个最佳匹配（排序之后取前k个），其中k是用户自己设定的
        good_matches = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
                good_matches.append((m.trainIdx, m.queryIdx))
        if len(good_matches) > 4:
            key_points1 = kp1['kp']
            key_points2 = kp2['kp']

            matched_kp1 = np.float32(
                [key_points1[i].pt for (_, i) in good_matches]
            )
            matched_kp2 = np.float32(
                [key_points2[i].pt for (i, _) in good_matches]
            )

            print('=======>Random sampling and computing the homography matrix!')
            homo_matrix, _ = cv2.findHomography(matched_kp1, matched_kp2, cv2.RANSAC, 4)
            return homo_matrix
        else:
            return None


class PasteTwoImages:
    def __int__(self):
        pass

    def __call__(self, img1, img2, homo_matrix):
        h1, w1 = img1.shape[0], img1.shape[1]
        h2, w2 = img2.shape[0], img2.shape[1]

        shft = np.array([[1.0, 0, w1], [0, 1.0, 0], [0, 0, 1.0]]) # 向右平移w1
        M = np.dot(shft, homo_matrix)  # 获取左边图像到右边图像的投影映射关系
        dst_corners = cv2.warpPerspective(img1, M, (w1 * 2, h1))  # 透视变换，新图像可容纳完整的两幅图
        dst_corners[0:h1, w1:w1 * 2] = img2  # 将第二幅图放在右侧

        return dst_corners


if __name__ == '__main__':

    img1_floder = '../image_dataset/img1.png'
    img2_floder = '../image_dataset/img2.png'
    image1 = cv2.imread(img1_floder)
    image2 = cv2.imread(img2_floder)
    sitchMatch = FindKeyPointsAndMatching()
    Kp1, Kp2 = sitchMatch.get_key_points(image1, image2)
    homo_matrix = sitchMatch.match(Kp1, Kp2)
    stitch_merge = PasteTwoImages()
    mergeImage = stitch_merge(image1, image2, homo_matrix)
    print('=======>Ready to show the stitched image!')
    cv2.namedWindow('img1', 0)
    cv2.imshow('img1', image1)
    cv2.namedWindow('img2', 0)
    cv2.imshow('img2', image2)
    cv2.namedWindow('output', 0)
    cv2.imshow('output', mergeImage)
    cv2.waitKey(50000)
    cv2.destroyAllWindows()

