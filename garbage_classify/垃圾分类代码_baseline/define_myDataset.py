import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np


# 数据集地址
data_path = os.getcwd() + '/train_data_v2/'


# 定义读取文件的格式
def default_loader(path):
    return Image.open(path).convert('RGB')


# 创建自己的类：MyDataset, 继承 Dataset类
class MyDataset(Dataset):
    def __init__(self, txt, data_path=None, transform=None, target_transform=None, loader=default_loader):
        super(MyDataset, self).__init__() # 对继承父类的属性初始化
        # 在__init__()方法中得到图像的路径，然后将图像路径组成一个数组
        file_path = data_path + txt
        file = open(file_path, 'r')
        imgs = []
        for line in file:
            line = line.split()
            # print(line[0].rstrip(','))  # img
            # print(line[1].rstrip('\n'))  # label
            imgs.append((line[0].rstrip(','), line[1].rstrip('\n')))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.data_path = data_path

    def __getitem__(self, index):
        # 按照索引读取每个元素的具体内容
        imgName, label = self.imgs[index]
        imgPath = self.data_path + imgName
        img = self.loader(imgPath)
        if self.transform is not None:
            img = self.transform(img)  # 数据标签转换为Tensor
            label = torch.from_numpy(np.array(int(label)))
        return img, label

    def __len__(self):
        # 数据集的图片数量
        return len(self.imgs)


# 图像的初始化操作
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])

test_transforms = transforms.Compose([
    transforms.RandomResizedCrop((227, 227)),
    transforms.ToTensor(),
])

# 数据集加载方式设置
train_data = MyDataset(txt='train.txt', data_path=data_path, transform=transforms.ToTensor())
test_data = MyDataset(txt='test.txt', data_path=data_path, transform=transforms.ToTensor())
# 调用DataLoader和数据集
train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)

# print('num_of_trainData:', len(train_data))
# print('num_of_testData:', len(test_data))
#
# for img, label in train_loader:
#     print(img.shape)
#     print(label)


