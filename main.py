#自行練習training
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(16*128, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x

class FC(torch.nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.l1 = torch.nn.Linear(784, 15)
        self.l2 = torch.nn.Linear(15, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)

        return x

def train_val(model1,optimizer_in,epoch):
    a=[]
    b1=[]
    max_acc=0.0
    f = open('{}.txt'.format(model1),'w')
    for i in range(epoch):
        model.train()
        total_correct=0
        total_data=0
        for iter,data in enumerate (train_loader):
            images, labels = data
            images=images.cuda()
            labels = labels.cuda()

            optimizer_in.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_in.step()

            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()
            total_data += labels.size(0)
        # print("train_epcoh",i,"ACC",total_correct/total_data)
        model.eval()
        total_correct=0
        total_data=0
        
        with torch.no_grad():
            
            
            for iter,data in enumerate (test_loader):

                images, labels = data
                images=images.cuda()
                labels = labels.cuda()

                optimizer_in.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                # loss.backward()
                # optimizer.step()

                _, predicted = torch.max(outputs.data, dim=1)
                total_correct += torch.eq(predicted, labels).sum().item()
                total_data += labels.size(0)

            acc=total_correct/total_data  
            acc *= 100
            if acc>max_acc:
                max_acc=acc
            else: max_acc=max_acc

            a.append(i)
            b1.append(max_acc)
            print("lr=",lr)
            d=float(max_acc)
            f.write(f"{d}\n")
    f.close()
            # print("val_epcoh",i,"ACC",acc,"max_acc",max_acc)

    return a,b1
if __name__ == "__main__":
    batch_size=64
    num_workers=4
    epoch=5
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='../dataset/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size,num_workers=4)

    test_dataset = datasets.MNIST(root='../dataset/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size,num_workers=4)



    # lr=0.001
    model=CNN()
    model1 = "CNN"
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    lr=0.0001
    optimizer_out = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    aa1,bb1=train_val(model1=model1,optimizer_in=optimizer_out,epoch=epoch)

    model=FC()
    model1 = "FC"
    model.cuda()
    lr=0.0001
    optimizer_out = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)
    aa2,bb2=train_val(model1=model1,optimizer_in=optimizer_out,epoch=epoch)


    
    # ---------------------------逐行讀取
    # f = open('{}.txt'.format(model))
    # a = f.readlines()
    # a = [int(i) for i in a ]
    # print(a)
    # f.close()
    # ---------------------------
    # 
    # text = f.read()
    # text=int(text)
    # a=[0,1,2,3]
    # print(text)
    # f.close()


    plt.title("MNIST CNN YCLin") # title
    plt.ylabel("Val_acc") # y label
    plt.xlabel("iteration") # x label
    plt.plot(aa1, bb1,color='r', label='CNN_acc')
    plt.plot(aa2,bb2,color='b', label='FC_acc')
    plt.legend()
    # plt.yticks(np.arange(98,100,step=0.2))
    plt.show()
