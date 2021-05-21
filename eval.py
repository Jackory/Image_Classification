import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image,ImageFile
import os
from shutil import copyfile
from collections import defaultdict
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

d = defaultdict(int)
def process_path(path, base='result'): # 将属于正类的图片文件名处理成：报名号+序号
    suffix = path.split('.')[-1]
    ID = path.split('\\')[-2]
    d[ID] += 1
    num = str(d[ID])
    return os.path.join(base, ID + '_' + num + '.' + suffix)


if not os.path.exists('result'):
    os.makedirs('result')

process = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

test = []
with open('test_data_paths.txt', 'r', encoding='utf-8') as f:
    paths = f.readlines()
    paths = [path.strip() for path in paths]
    for pth in paths:
        img = Image.open(pth).convert('RGB')
        tensor = process(img)
        test.append((tensor, pth))

test_loader = DataLoader(ImageDataset(test), batch_size = 100, shuffle=False)

device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('models/final2.pt')
model.to(device)

with torch.no_grad():
    model.eval()
    for (inputs, path) in tqdm(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        ret, predictions = torch.max(outputs.data, 1)
        for i, pred in enumerate(predictions.data):
            if pred == 1:
                target_path = process_path(path[i])
                copyfile(path[i], target_path)