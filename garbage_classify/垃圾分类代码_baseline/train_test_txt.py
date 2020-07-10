import os

# 数据集地址, 生成训练集和测试集
data_path = os.getcwd() + '/train_data_v2/'
print(data_path)


data_list = os.listdir(data_path)
train_names = open('./train_data_v2/train.txt', 'a')
test_names = open('./train_data_v2/test.txt', 'a')
# print(len(data_list))
print(len(data_list)/2)  # 14683张图片，其中一万张图片为训练图片
i = 1
for name in data_list:
    if '.txt' in name:
        file = open(data_path + name, 'r')
        data = file.read()
        file.close()
        data = data + '\n'
        if i <= 10000:
            train_names.write(data)
            i += 1
        else:
            test_names.write(data)
            i += 1
        # pass
        # print(data_name)
train_names.close()
test_names.close()

'''
file = open(data_path + 'train.txt', 'r')
i = 1
for line in file:
    line = line.split()
    print(line[0].rstrip(','))  # img
    print(line[1].rstrip('\n'))  # label
    if i == 1:
        break
'''