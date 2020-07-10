# -*- coding: utf-8 -*-
import numpy as np
import cv2
import random
import os

# calculate means and std
# 数据集地址
data_path = os.getcwd() + '/train_data_v2/'
path = data_path + 'train.txt'
means = [0, 0, 0]
stdevs = [0, 0, 0]

index = 1
num_imgs = 0
with open(path, 'r') as f:
    lines = f.readlines()
    # random.shuffle(lines)
    # print(lines)
    for line in lines:
        line = line.split()
        print(line[0].rstrip(','))
        print('{}/{}'.format(index, len(lines)))
        index += 1
        imgPath = os.path.join(data_path, line[0].rstrip(','))

        num_imgs += 1
        img = cv2.imread(imgPath) # cv2默认为bgr顺序
        img = np.asarray(img)
        # print(img.shape)
        img = img.astype(np.float32) / 255.
        for i in range(3):
            means[i] += img[:, :, i].mean()
            stdevs[i] += img[:, :, i].std()
print(num_imgs)
means.reverse()
stdevs.reverse()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

'''
normMean = [0.54483902 0.50552646 0.45878888]
normStd = [0.20683282 0.21217371 0.21958046]
transforms.Normalize(normMean = [0.54483902 0.50552646 0.45878888], normStd = [0.20683282 0.21217371 0.21958046])
'''