########dericative rules

####1.Basic Rule
####2.Product rule
#(fg)' = f'g + fg'
####3.Quotient rule
#(f/g)'=(f'g + fg')/g*g
#e.g. softmax
####4.Chain Rule
#神经网络中有激活函数，求导非常麻烦，引入中间变量多次求导，然后可以简化计算
import torch

x=torch.tensor(1.)
w1=torch.tensor(2.,requires_grad=True)
b1=torch.tensor(1.)

w2=torch.tensor(2.,requires_grad=True)
b2=torch.tensor(1.)

y1=x*w1+b1
y2=y1*w2+b2

dy2_dy1=torch.autograd.grad(y2,[y1],retain_graph=True)[0]
dy1_dw1=torch.autograd.grad(y1,[w1],retain_graph=True)[0]
dy2_dw1=torch.autograd.grad(y2,[w1],retain_graph=True)[0]

c=dy2_dy1*dy1_dw1
print(c)
print(dy2_dw1)
