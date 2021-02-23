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
LeNet: LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.
'''
class LeNet(nn.Module):
    '''
    Build LeNet Model
    '''
    def __init__(self, args):
        super(LeNet, self).__init__() 

 
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)
            nn.Conv2d(args.num_channels, 32, 5, 1, 2),   # padding=2
            nn.ReLU(),                  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)
        )

     
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),  # input_size=(6*14*14)  
            nn.ReLU(),            # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)    # output_size=(16*5*5)
        )

 
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  
            nn.ReLU()                    
        )

        
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

 
        self.fc3 = nn.Linear(84, 62)

  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
 
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
