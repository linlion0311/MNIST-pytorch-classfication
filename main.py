# 自行練習training 使用CNN及FCN
import cv2
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms
from random import random
from model.CNN import CNN


def train_val(model, criterion, optimizer, lr_scheduler):
    model.train()
    total_correct, total_data = 0, 0
    loop = tqdm(enumerate(train_loader), total=len(
        train_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
    for iter, (images, labels) in loop:
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

        _, predicted = torch.max(outputs.data, dim=1)
        total_correct += torch.eq(predicted, labels).sum().item()
        total_data += labels.size(0)

    train_acc = (total_correct/total_data)*100
    lr_scheduler.step()

    model.eval()
    total_correct, total_data, max_acc = 0, 0, 0.0
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(
            test_loader), desc=f'Epoch {epoch+1}/{args.epochs}')
        for iter, (images, labels) in loop:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()
            total_data += labels.size(0)

        val_acc = (total_correct/total_data)*100
        if val_acc >= max_acc:
            max_acc = val_acc
        else:
            max_acc = max_acc
    return train_acc, val_acc, max_acc


# 當前是以MNIST為data訓練
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int,
                        default=32, help="batch size for dataloader")
    parser.add_argument("--num_workers", type=int,
                        default=8, help="max dataloader workers")
    parser.add_argument("--epochs", type=int, default=20, help="train epochs")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root='./data/', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=args.batch_size, num_workers=args.num_workers)

    test_dataset = datasets.MNIST(
        root='./data/', train=False, download=False, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False,
                             batch_size=args.batch_size, num_workers=args.num_workers)
    model = CNN().cuda()
    f = open('{}.txt'.format("ResNet18"), 'w')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [5, 10, 20], gamma=0.1)

    fig = plt.figure()
    plt.ion()
    epoch_list, train_acc_list, val_acc_list = [], [], []

    for epoch in range(args.epochs):

        train_acc, val_acc, max_acc = train_val(
            model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler)
        epoch_list.append(epoch)
        val_acc_list.append(val_acc)
        train_acc_list.append(train_acc)

        print(optimizer.state_dict()['param_groups'][0]['lr'])

        f.write('{}\n'.format(val_acc))
        fig.clf()
        plt.title("CIFAR10 Classfication SUNNY")  # title
        plt.xlabel("epoch")  # x label
        plt.ylabel("train_acc")  # y label
        plt.subplot(121)
        plt.plot(epoch_list, train_acc_list, label='resnet18')
        # if epoch == epoch:
        plt.legend()
        plt.subplot(122)
        plt.plot(epoch_list, val_acc_list, label='resnet18_val')
        # if epoch == epoch:
        plt.legend()
        plt.pause(1)

    plt.ioff()
    plt.savefig('runs/trian/result.png')
    plt.show()

    # import matplotlib.pyplot as plt
    # a = 1
    # while a <= 10:
    #     b = plt.scatter(a, a ** 2, color='r', label="A")
    #     plt.pause(0.1)
    #     if a == 1:
    #         plt.legend()
    #     a = a + 1
    #     plt.show()
