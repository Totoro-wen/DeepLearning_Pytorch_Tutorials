###分类+实现ResNet
#1.CIFAR10
#10大类
#加拿大

##CIFAR100
#将10类细换分为100类

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from lenet5 import Lenet5
from torch import nn
import  torch.optim as optim


def main():
    bachsz=32
    cifar_train = datasets.CIFAR10('cifar',True , transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)#一次加载一张
    cifar_train=DataLoader(cifar_train, batch_size=bachsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)  # 一次加载一张
    cifar_test = DataLoader(cifar_test, batch_size=bachsz, shuffle=True)

    x,label = iter(cifar_train).nest()
    print('x:',x.shape, 'label:',label.shape)


    device = torch.device('cuda')
    model = Lenet5.to(device)
    criteon = nn.CrossEntropyLoss().to(device)#包含softmax
    optimizer=optim.Adam(model.parameter(),lr=1e-3)
    print(model)
    for epoch in range(1000):
        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            #[b,3,32,32]
            #[b]
            x, label=x.to(device),label.to(device)
            logits =model(x)
            #logits:[b,10]
            #label:[10]
            #loss :tensor scalar长度为0的标量
            loss=criteon(logits,label)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #
        print('epoch:',epoch, 'loss',loss.item())
        #loss.item()转换为numpy打印,最后一个batch的loss

        model.eval()#
        with torch.no_grad():#不需要backprop
            #test
            total_correct = 0
            total_num =0
            for x,label in cifar_test:
                # [b,3,32,32]
                # [b]
                x, label = x.to(device), label.to(device)

                #[b,10]
                logits = model(x)
                # [b]
                pred=logits.argmax(dim=1)#返回数据上dim=1上最大值的索引
                total_correct += pred.eq(pred, label).float().sum().item()
                total_num +=x.size(0)
            acc=total_correct/total_num
            print('epoch:',epoch,'acc:', acc)


if __name__=='__main__':
    main()












