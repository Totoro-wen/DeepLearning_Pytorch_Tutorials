###全连接层
### I know nothing


###be Practial

#1.nn.linear
# import torch
# import torch.nn as nn
#
# from torch import functional as F
#
#
# x=torch.randn(1,784)
# print(x.shape)
#
# #                 in,out
# layer1=nn.linear(784,200)
# layer2=nn.linear(200,200)
# layer3=nn.linear(200,10)
#
# x=layer1(x)
# #能用relu就一定用,sigmoid在(RGB的像素重建)情况下使用
# x=F.relu(x,inplace=True)
# print(x.shape)
#
# x=layer2(x)
# x=F.relu(x,inplace=True)
# print(x.shape)
#
# x=layer3(x)
# x=F.relu(x,inplace=True)
# print(x.shape)

#2.concisely
##inherit from nn.Module
##init layer in __init__
##implement forward()
# #autograd 能保存向前计算的图
# class MLP(nn.Module):
#     def __init__(self):
#         super(MLP,self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(784, 200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, 200),
#             nn.ReLU(inplace=True),
#             nn.Linear(200, 10),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         x=self.model(x)#使用了model.forward,继承了nn.Module
#         return x

#3.nn.ReLU v.s. F.relu()
##class-style API,
### eg, nn.Linear,必须先实例化,在调用,不能私自访问参数，必须使用.parameter()
##function-style API,
### eg. F.relu,仅仅只是用GPU加速



#4.train
import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms


batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)






class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #没有初始化,因为函数API自带初始化,我们也可以自己写





































