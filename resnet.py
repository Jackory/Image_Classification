
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

import os
 
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

 
batch_size = 32
num_classes = 2
 
data = {
    'train': datasets.ImageFolder(root='train', transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root='valid', transform=image_transforms['valid'])
}
 
 
train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
 
train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
 
resnet50 = models.resnet50(pretrained=True)


for param in resnet50.parameters():
    param.requires_grad = False


fc_inputs = resnet50.fc.in_features
resnet50.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2),
    nn.LogSoftmax(dim=1)
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)

loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet50.parameters())


def train_and_valid(model, loss_function, optimizer, epochs=25):
    history = []
    best_acc = 0.0
    best_epoch = 0
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
 
        model.train()
 
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        train_precision = 0.0
        valid_precision = 0.0
        train_recall = 0.0
        valid_recall = 0.0
 
        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            inputs = inputs.to(device)
            labels = labels.to(device)
 
            #因为这里梯度是累加的，所以每次记得清零
            optimizer.zero_grad()
 
            outputs = model(inputs)
 
            loss = loss_function(outputs, labels)
 
            loss.backward()
 
            optimizer.step()
 
            train_loss += loss.item() * inputs.size(0)
 
            ret, predictions = torch.max(outputs.data, 1)
            correct = labels.data.view_as(predictions)

            correct_counts = predictions.eq(correct)
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            precision = precision_score(correct, predictions.data)
            recall = recall_score(correct, predictions.data)
            train_acc += acc.item() * inputs.size(0)
            train_precision += precision * inputs.size(0)
            train_recall += recall * inputs.size(0)
 
        with torch.no_grad():
            model.eval()
 
            for j, (inputs, labels) in enumerate(tqdm(valid_data)):
                inputs = inputs.to(device)
                labels = labels.to(device)
 
                outputs = model(inputs)
 
                loss = loss_function(outputs, labels)
 
                valid_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct = labels.data.view_as(predictions)

                correct_counts = predictions.eq(correct)
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                precision = precision_score(correct, predictions.data)
                recall = recall_score(correct, predictions.data)
                valid_acc += acc.item() * inputs.size(0)
                valid_precision += precision * inputs.size(0)
                valid_recall += recall * inputs.size(0)
 
        avg_train_loss = train_loss/train_data_size
        avg_train_acc = train_acc/train_data_size
        avg_train_precision = train_precision/train_data_size
        avg_train_recall = train_recall/train_data_size
 
        avg_valid_loss = valid_loss/valid_data_size
        avg_valid_acc = valid_acc/valid_data_size
        avg_valid_precision = valid_precision/valid_data_size
        avg_valid_recall = valid_recall/valid_data_size
 
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
 
        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1
 
        epoch_end = time.time()
 
        print("Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Precison: {:.4f}%, Recall: {:.4f}%,\n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Precison: {:.4f}%, Recall: {:.4f}%, Time: {:.4f}s".format(
            epoch+1, avg_valid_loss, avg_train_acc*100, avg_train_precision*100, avg_train_recall*100, avg_valid_loss, avg_valid_acc*100, avg_valid_precision*100, avg_valid_recall*100,epoch_end-epoch_start
        ))
        print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
 
        torch.save(model, 'models/'+'_model_'+str(epoch+1)+'.pt')
    return model, history


num_epochs = 30
trained_model, history = train_and_valid(resnet50, loss_func, optimizer, num_epochs)
torch.save(history, 'models/'+'_history.pt')
 
history = np.array(history)
plt.figure()
plt.plot(history[:, 0:2])
plt.legend(['Tr Loss', 'Val Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.savefig('_loss_curve.png')
plt.show()
 
plt.figure()
plt.plot(history[:, 2:4])
plt.legend(['Tr Accuracy', 'Val Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('_accuracy_curve.png')
plt.show()
