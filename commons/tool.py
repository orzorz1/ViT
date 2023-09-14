from torch.nn import functional as F
import torch
import os
import random


def label_to_onehot(img):
    out = F.one_hot(img)
    out = out.cpu().numpy()
    out = out.transpose(3, 0, 1, 2)
    out = torch.tensor(out)

    return out


def listdir(path):
    file_names = os.listdir(path)
    file_names.sort()
    list = []
    for name in file_names:
        f_format = "nii.gz"
        new_name = path + '/' + name  # 新名字加上序号
        list.append(new_name)
    return list

def shuffle(x, y, rand = None, use_cuda = False):
    '''
    :param x、y:需要打乱的tensor
    :param rand: [0.0, 1.0)的随机数
    :param use_cuda: tensor是否使用gpu运算
    :return: 以相同方式打乱的两个tensor
    '''
    if rand is None:
        rand = random.random() #random=random.random
    #转成numpy
    if torch.is_tensor(x)==True:
        if use_cuda==True:
           x=x.cpu().detach().numpy()
        else:
           x=x.numpy()
    if torch.is_tensor(y) == True:
        if use_cuda==True:
           y=y.cpu().detach().numpy()
        else:
           y=y.numpy()
    #开始随机置换
    for i in range(len(x)):
        j = int(rand * (i + 1))
        if j<=len(x)-1:#交换
            x[i],x[j]=x[j],x[i]
            y[i],y[j]=y[j],y[i]
    #转回tensor
    if use_cuda == True:
        x=torch.from_numpy(x).cuda()
        y=torch.from_numpy(y).cuda()
    else:
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
    return x,y