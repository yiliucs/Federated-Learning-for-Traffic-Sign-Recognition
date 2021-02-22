#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Nets import MLP, CNNMnist, CNNCifar, LeNet
from load_data import Traffic

import loading_data as dataset

def get_train_valid_loader(data_dir,
                           batch_size,
                           num_workers=0,
                           ):
    # Create Transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])

    # Create Datasets
    dataset_train = dataset.BelgiumTS(
        root_dir=data_dir, train=True,  transform=transform)
    dataset_test = dataset.BelgiumTS(
        root_dir=data_dir, train=False,  transform=transform)

    # Load Datasets
    return dataset_train, dataset_test

def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    #parser.add_argument('--outf', default='./save/', help='folder to output images and model checkpoints')  # 模型保存路径
    #parser.add_argument('--net', default='./save/net.pth', help="path to netG (to continue training)")  # 模型加载路径
    #opt = parser.parse_args()  # 解析得到你在路径中输入的参数，比如 --outf 后的"model"或者 --net 后的"model/net_005.pth"，是作为字符串形式保存的

    # Load training and testing datasets.
    torch.manual_seed(args.seed)

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
        img_size = dataset_train[0][0].shape
    elif args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader('/home/liuyi/Documents/federated-learning-master/federated-learning-master/data',
                                                                         batch_size=32, num_workers=0)

        # 定义训练数据集(此处是加载MNIST手写数据集)
       # transform = transforms.Compose([
        #    transforms.Resize((28, 28)),
        #    transforms.CenterCrop(28),
        #    transforms.ToTensor()])

       # dataset_train = Traffic(
       #     root=train_data_dir,  # 如果从本地加载数据集，对应的加载路径
       #     train=True,  # 训练模型
        #    download=True,  # 是否从网络下载训练数据集
        #    transform=transform  # 数据的转换形式
       # )

        # 定义训练批处理数据

    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'LeNet' and args.dataset == 'traffic':
        net_glob = LeNet(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=10, shuffle=True, num_workers=0)
    #train_loader = DataLoader(
    #    dataset_train,  # 加载测试集
    #    batch_size=3,  # 最小批处理尺寸
    #    shuffle=True,  # 标识进行数据迭代时候将数据打乱
   # )

    list_loss = []
    net_glob.train()
    for epoch in range(args.epochs):
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

    # plot loss
    plt.figure()
    plt.plot(range(len(list_loss)), list_loss)
    plt.xlabel('epochs')
    plt.ylabel('train loss')
    plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))

    # testing
    if args.dataset == 'mnist':
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        test_loader = DataLoader(dataset_test, batch_size=10, shuffle=False)
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
        test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
    elif args.dataset == 'traffic':
        dataset_train, dataset_test = get_train_valid_loader(
            '/home/liuyi/Documents/federated-learning-master/federated-learning-master/data',
            batch_size=32, num_workers=0)
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=10, shuffle=False, num_workers=0)
        # 定义测试数据集
        #transform = transforms.Compose([
        #    transforms.Resize((28, 28)),
        #    transforms.CenterCrop(28),
        #    transforms.ToTensor()])

        #dataset_test = Traffic(
         #   root=test_data_dir,  # 如果从本地加载数据集，对应的加载路径
         #   train=True,  # 训练模型
         #   download=True,  # 是否从网络下载训练数据集
          #  transform=transform  # 数据的转换形式
        #)
        # 定义测试批处理数据
        #test_loader = DataLoader(
        #    dataset_test,  # 加载测试集
        #    batch_size=10,  # 最小批处理尺寸
         #   shuffle=False,  # 标识进行数据迭代时候不将数据打乱
        #)
    else:
        exit('Error: unrecognized dataset')

    print('test on', len(dataset_test), 'samples')
    test_acc, test_loss = test(net_glob, test_loader)
