##########数学操作基础知识#####
####1.add/minus/multiply/divide
####2.matmul
####3.pow
####4.sqrt/rsqrt
####5.round

import torch
#######basic
a=torch.rand(3,4)
b=torch.rand(4)
# print(a)
# print(b)
# print(a+b)
# c=torch.add(a,b)
# print(c)
# d=torch.all(torch.eq(a-b,torch.sub(a,b)))
# print(d)
# d=torch.all(torch.eq(a*b,torch.mul(a,b)))
# print(d)
# d=torch.all(torch.eq(a/b,torch.div(a,b)))
# print(d)
#######2.element-wise元素相乘
#####matrix mul
### torch.mm(only for 2d)
### torch.matmul(推荐)
### @==torch.matmul
a=torch.tensor([[3.,3.],[3.,3.]])
# print(a)
b=torch.ones(2,2)
# c=torch.mm(a,b)
# print(c)
# d=torch.matmul(a,b)
# print(d)
# print(a@b)
####eg. in cnn
# a=torch.rand(4,784)
# x=torch.rand(4,784)
# w=torch.rand(512,784)
# print((x@w.t()).shape)
#######>2d tensor matmul
# a=torch.rand(4,3,28,64)
# b=torch.rand(4,3,64,32)
# print(torch.matmul(a,b).shape)#需要前两维一致性batch_size,channel
# b=torch.rand(4,1,64,32)
# print(torch.matmul(a,b).shape)#boradcasting
########power
a=torch.full([2,2],3)
aa=a.pow(2)#==  aa==a**2
#####exp ,log默认以e为底
a=torch.exp(torch.ones(2,2))
print(a)
print(torch.log(a))
####Approximation
a=torch.tensor(3.14)
print(a.floor())#向下取整
print(a.ceil())#向上取证
print(a.trunc())#取整数部分
print(a.frac())#取小数部分
print(a.round())#四舍五入
#############clamp裁剪
#######gradient clipping
#######(min)
#######(min.max)
#######打印梯度zhi:w.grad.norm(2)<10#梯度的L2范数
grad=torch.rand(2,3)*15
print(grad)
print(grad.max())
print(grad.median())
print(grad.clamp(10))
print(grad.clamp(1,10))





















