import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet50
from define_myDataset import MyDataset
from torch.utils.data import Dataset, DataLoader

# 数据集地址
data_path = os.getcwd() + '/train_data_v2/'

# 预处理的设置
# 图片转化为resnet50规定的图片大小
# 归一化是减去均值，除以方差
# 把 numpy array 转化为 tensor 的格式
my_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.545, 0.506, 0.459], [0.207, 0.212, 0.220])])

# 数据集加载方式设置
train_data = MyDataset(txt='train.txt', data_path=data_path, transform=my_tf)
test_data = MyDataset(txt='test.txt', data_path=data_path, transform=my_tf)

# 调用DataLoader和数据集
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=2)

# 使用resnet50
my_resnet50 = resnet50(pretrained=True)

# 固定网络框架全连接层之前的参数
for param in my_resnet50.parameters():
    param.requires_grad = False

# 将resnet50最后一层输出的类别数，改为垃圾分类数据集的类别数（40）
in_f = my_resnet50.fc.in_features
my_resnet50.fc = nn.Linear(in_f, 40)

# 超参数设置
learn_rate = 0.001
num_epoches = 2
# 多分类损失函数，使用默认值
criterion = nn.CrossEntropyLoss()
# 梯度下降，求解模型最后一层参数
optimizer = optim.SGD(my_resnet50.fc.parameters(), lr=learn_rate, momentum=0.9)
# 判断使用CPU还是GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 训练阶段
my_resnet50.to(device)
my_resnet50.train()
for epoch in range(num_epoches):
    print(f"epoch: {epoch+1}")
    for idx, (img, label) in enumerate(train_loader):
        images = img.to(device)
        labels = label.to(device)
        output = my_resnet50(images)
        loss = criterion(output, labels)
        loss.backward()  # 损失反向传播
        optimizer.step()  # 更新梯度
        optimizer.zero_grad()  # 梯度清零
        if idx % 100 == 0:
            print(f"current loss = {loss.item()}")


# 测试阶段
my_resnet50.to(device)
my_resnet50.eval()  # 把训练好的模型的参数冻结
total, correct = 0, 0
for img, label in test_loader:
    images = img.to(device)
    labels = label.to(device)
    #print("label: ",labels)
    output = my_resnet50(images)
    #print("output:", output.data.size)
    _, idx = torch.max(output.data, 1) # 输出最大值的位置
    #print("idx: ", idx)
    total += labels.size(0) # 全部图片
    correct += (idx == labels).sum() # 正确的图片
    #print("correct_num: %f",correct)
print("correct_num: ", correct)
print("total_image_num: ", total)
print(f"accuracy:{100.*correct/total}")