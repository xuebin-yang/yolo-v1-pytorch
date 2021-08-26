# -*- coding: utf-8 -*-
"""
@Time          : 2020/08/07 15:07
@Author        : FelixFu
@File          : train.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :
"""
import warnings
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models

from models.vgg_yolo import vgg16_bn
from models.resnet_yolo import resnet50
from models.yoloLoss import yoloLoss
from utils.dataset import yoloDataset

warnings.filterwarnings('ignore')

# 设置GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 判断GPU是否可用
use_gpu = torch.cuda.is_available()

if __name__ == '__main__':
    # 数据文件
    file_root = 'datasets'

    # 超参数
    learning_rate = 0.001
    num_epochs = 100
    batch_size = 2

    # checkpoints
    resume = False

    # ---------------------数据读取---------------------
    train_dataset = yoloDataset(root=file_root, list_file='voc2007.txt', train=True,
                                transform=[transforms.ToTensor()])
    # train_dataset = yoloDataset(root=file_root, list_file=['voc12_trainval.txt','voc07_trainval.txt'],
    #                           train=True,transform = [transforms.ToTensor()] )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    test_dataset = yoloDataset(root=file_root, list_file='voc2007test.txt', train=False,
                               transform=[transforms.ToTensor()])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print('the dataset has %d images' % (len(train_dataset)))
    print('the batch_size is %d' % batch_size)


    # ---------------------网络选择---------------------
    use_resnet = True
    if use_resnet:
        net = resnet50()
        print(net)
    else:
        net = vgg16_bn()

    if resume:
        print("loading weight from checkpoints/best.pth")
        net.load_state_dict(torch.load('checkpoints/best.pth'))
    else:
        print('loading pre-trined model ......')
        if use_resnet:
            resnet = models.resnet50(pretrained=True)
            new_state_dict = resnet.state_dict()
            dd = net.state_dict()
            for k in new_state_dict.keys():
                # print(k)
                if k in dd.keys() and not k.startswith('fc'):
                    # print('yes')
                    dd[k] = new_state_dict[k]
            net.load_state_dict(dd)
        else:
            vgg = models.vgg16_bn(pretrained=True)
            new_state_dict = vgg.state_dict()
            dd = net.state_dict()
            for k in new_state_dict.keys():
                print(k)
                if k in dd.keys() and k.startswith('features'):
                    print('yes')
                    dd[k] = new_state_dict[k]
            net.load_state_dict(dd)

    if use_gpu:
        print('this computer has gpu %d and current is %s' % (torch.cuda.device_count(),
              torch.cuda.current_device()))
        net.cuda()


    # ---------------------损失函数---------------------
    criterion = yoloLoss(7, 2, 5, 0.5)

    # ---------------------优化器----------------------

    # different learning rate
    params = []
    params_dict = dict(net.named_parameters())
    for key, value in params_dict.items():
        if key.startswith('features'):
            params += [{'params': [value], 'lr':learning_rate*1}]
        else:
            params += [{'params': [value], 'lr':learning_rate}]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=1e-4)


    # ---------------------训练---------------------
    logfile = open('checkpoints/log.txt', 'w')
    num_iter = 0
    best_test_loss = np.inf

    for epoch in range(num_epochs):
        # train
        net.train()
        if epoch == 30:   # 这是个全局参数
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
        print('Learning Rate for this epoch: {}'.format(learning_rate))

        total_loss = 0.

        for i, (images, target) in enumerate(train_loader):
            if use_gpu:
                images, target = images.cuda(), target.cuda()
                print("target.shape = ", target.shape)    # [2, 14, 14, 30]
                print("images.shape = ", images.shape)    # [2, 3, 448, 448]

            pred = net(images)     # [2, 14, 14, 30]
            print("pred.shape = ", pred.shape)
            loss = criterion(pred, target)
            total_loss += loss.data.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 5 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f'
                            % (epoch+1, num_epochs, i+1, len(train_loader), loss.item(), total_loss / (i+1)))
                num_iter += 1

        # validation
        validation_loss = 0.0
        net.eval()
        for i, (images, target) in enumerate(test_loader):
            if use_gpu:
                images, target = images.cuda(), target.cuda()

            pred = net(images)
            loss = criterion(pred, target)
            validation_loss += loss.item()
        validation_loss /= len(test_loader)

        if best_test_loss > validation_loss:
            best_test_loss = validation_loss
            print('get best test loss %.5f' % best_test_loss)
            torch.save(net.state_dict(), 'checkpoints/best.pth')
        logfile.writelines(str(epoch) + '\t' + str(validation_loss) + '\n')
        logfile.flush()


