#创建tensor
#2019-04-10
import numpy as np
import torch


##import from numpy
# a=np.array([2,3.3])
# print(torch.from_numpy(a))#将参数作为numpy导入
#
# a=np.ones([2,3])
# print(torch.from_numpy(a))
#
# ###import from list (数据两不是很大)
# a=torch.tensor([2.,3.2])#numpy\list\现成的数据
# #区别：FloatTensor（shape）
# #Tensor()数据维度
# #Tensor=FloatTensor
#
#
#
# print(a)
# a=torch.FloatTensor([2.,3.2])#不要使用
# print(a)    #tensor([2.0000, 3.2000])
#
# ###uninitialized
# # torch.empty()
# # torch.FloatTensor(d1,d2,d3)
# # torch.IntTensor(d1,d2,d3)
#
# ##########set default type
# torch.tensor([1.2,3]).type()#torch.FloatTensor
#
# torch.set_default_tensor_type(torch.DoubleTensor)
#
# torch.tensor([1.2, 3]).type()#torch.DoubleTensor
########rand/rand_like,randint
# a=torch.rand(3,3)
# print(a)
# print(torch.rand_like(a))
# #(min,max)--->[1,10)
# a=torch.randint(1,10,[3 ,3])

######randn
# a=torch.randn(3,3)#~N(0,1)
# print(a)
# a=torch.normal(mean=torch.full([10],0),std=torch.arange(1,0,-0.1))
# print(a)
# a=torch.full([2,3],7)
# print(a)
# a=torch.full([2,3],7)#shape 标量
# print(a)
########arange/range
# a= torch.arange(0,10)
# print(a)#tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# a= torch.arange(0,10,2)
# print(a)#tensor([0, 2, 4, 6, 8])
# a=torch.range(0,10)
# print(a)
########linspace/logspace
# a=torch.linspace(0,10,steps=4)
# print(a)#tensor([ 0.0000,  3.3333,  6.6667, 10.0000])
# a=torch.logspace(0,-1,steps=11)
# print(a)
# a=torch.ones(3,3)
# # a=torch.zeros(3,3)
# # a=torch.eye(3,3)#对角矩阵
# a=torch.ones_like(a)
# print(a)
#########randperm()->shuffle
a=torch.randperm(9)
print(a)

























