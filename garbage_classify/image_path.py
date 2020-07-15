import os

'''
以下三个文件在同一个路径下：
- train_data文件夹（扩增的图片，作为测试集，4165张）
- train_data_v2文件夹（作为训练集，14683张） 也可以分出 train  val 
- image_path.py，生成 new_train.txt, new_test.txt
warning: train_data_v2文件夹里面有train.txt, text.txt，请讲这两个文件移除再执行该文件。
否则写入new_train.txt文件中会有train.txt，test.txt的内容
'''

cmd_path = os.getcwd()
print(cmd_path)
#  /Users/apple/Desktop/后厂理工/cv_projects/garbage_classify
dataset_name = 'train_data_v2' # train dataset
# dataset_name = 'train_data'  # test dataset
data_path = cmd_path + '/' + dataset_name + '/'
data_list = os.listdir(data_path)
train_file = open('./new_train.txt', 'a')
# test_file = open('./new_test.txt', 'a')

for name in data_list:
    if '.txt' in name:
        file = open(data_path + name, 'r')
        data = file.read()
        file.close()
        new_data = os.path.join(dataset_name, data) + '\n'
        train_file.write(new_data)
        # test_file.write(data)
train_file.close()
# test_file.close()






