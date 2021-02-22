#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

'''
定义LeNet神经网络，进一步的理解可查看Pytorch入门，里面很详细，代码本质上是一样的，这里做了一些封装
'''
class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''
    def __init__(self, args):
        super(LeNet, self).__init__()  # 这一个是python中的调用父类LeNet的方法，因为LeNet继承了nn.Module，如果不加这一句，无法使用导入的torch.nn中的方法，这涉及到python的类继承问题，你暂时不用深究

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)：输入层图片的输入尺寸，我看了那个文档，发现不需要天，会自动适配维度
            nn.Conv2d(args.num_channels, 32, 5, 1, 2),   # padding=2保证输入输出尺寸相同：采用的是两个像素点进行填充，用尺寸为5的卷积核，保证了输入和输出尺寸的相同
            nn.ReLU(),                  # input_size=(6*28*28)：同上，其中的6是卷积后得到的通道个数，或者叫特征个数，进行ReLu激活
            nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)：经过池化层后的输出
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),  # input_size=(6*14*14)：  经过上一层池化层后的输出,作为第二层卷积层的输入，不采用填充方式进行卷积
            nn.ReLU(),            # input_size=(16*10*10)： 对卷积神经网络的输出进行ReLu激活
            nn.MaxPool2d(2, 2)    # output_size=(16*5*5)：  池化层后的输出结果
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 62)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
