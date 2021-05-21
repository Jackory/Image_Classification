import os
from PIL import Image,ImageFile
from shutil import copyfile

test_data_paths = []
for root, dirs, files in os.walk('综合评价'):
    for file in files:
        temp = root.split('\\')[-1]
        if str.isdigit(temp) and '身份证' not in file: # 从测试集中筛选数据
            path = os.path.join(root, file)
            test_data_paths.append(path)

with open('test_data_paths.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(test_data_paths))

