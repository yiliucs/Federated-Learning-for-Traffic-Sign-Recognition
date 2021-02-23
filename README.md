# Federated-Learning-for-Traffic-Sign-Recognition
## Abstract
In this project, we build a simple federated deep learning model (i.e., [LeNet](https://ieeexplore.ieee.org/abstract/document/726791) model) that classifies 62 differents Belgian Traffic Signs.

---

## Dataset 
* You can download dataset from [here](https://btsd.ethz.ch/shareddata/).
* You also can download dataset from [Google Drive](https://drive.google.com/file/d/1K84SSq4jYQu5N1ETqvkRjkStJk_rjMjy/view?usp=sharing).
* [Baidu Yun](https://pan.baidu.com/s/1JYWEFYFJCSRsVPmBfkauPQ), pwd: **wqtv**.

![](https://codimd.xixiaoyao.cn/uploads/upload_96f7deca3d7c7b01d9fb03feaf15d1bc.png)

---

## Model
In this project, we use LeNet model to classify 62 differents Belgian Traffic Signs. For more detials can be seen in [[1]](https://ieeexplore.ieee.org/abstract/document/726791).

[1]: In Proceedings of the IEEE [[bibtex]](https://ieeexplore.ieee.org/abstract/document/726791)

```
@ARTICLE{726791,
  author={Y. {Lecun} and L. {Bottou} and Y. {Bengio} and P. {Haffner}},
  journal={Proceedings of the IEEE}, 
  title={Gradient-based learning applied to document recognition}, 
  year={1998},
  volume={86},
  number={11},
  pages={2278-2324},
  doi={10.1109/5.726791}}
```

![](https://codimd.xixiaoyao.cn/uploads/upload_95b7ddf46b1a3671a02fb5b210c999f4.png)


Code: [PyTorch implementation of LeNet-5 with live visualization](https://github.com/activatedgeek/LeNet-5).

> Please note that **you can replace other convolutional neural networks such as ResNet and AlexNet model**. You just need to **pay attention to the input size** of different convolutional neural networks.

---

## How to run
* Setp 1: Download the dataset via the above links.
* Setp 2: If you want to run the centralized LeNet model, please via the following command line:
```python=
python main_nn.py --dataset traffic --iid --num_channels 3 --model LeNet --epochs 500 --gpu -1
```
* If you want to run the federated LeNet model, please via the following command line:
 ```python=
python main_fed.py --dataset traffic --iid --num_channels 3 --model LeNet --epochs 1000 --gpu -1
```

> For the following command:
 ```python=
dataset_train, dataset_test = get_train_valid_loader('/home/liuyi/Documents/federated-learning-master/federated-learning-master/data', batch_size=32, num_workers=0)
```
> **You need to use your data path.**
* Setp 3: For centralized learning, you can access the folder *log/...*. For federated learning, you can access the folder *save/...*.

---

## Experimental results
### Centralized Learning
![](https://codimd.xixiaoyao.cn/uploads/upload_47bf4011253f601f9d2b717bf3916d92.png)
---
### Federated Learning
![](https://codimd.xixiaoyao.cn/uploads/upload_ebf84a98c60c4ad5803f642b68b974c4.png)


