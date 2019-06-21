#############operation
#View/==reshape
#squeeze(压缩)/unsqueeze
#Transpose/t/permute
#Expand/repeat
#######################
import torch

a=torch.rand(4,1,28,28)
######view reshape
#lsot dim infromation
# print(a.view(4,28*28))#保证size不变prod(a.size)=prod(a'.size)
# print(a.view(4,28*28).shape)
# print(a.view(4*28,28))#理解方式不一样，只关注28列
#
# b=a.view(4,784)
# b.view(4,28,28,1)#logic bug
#######unsqueeze不会改变数据本身，只是给数据增加了一个组别
# print(a.shape)
#插入数计算
#【-a.dim()-1,a.dim+1】
# print(a.unsqueeze(0).shape)#在索引0前插入
# print(a.unsqueeze(-1).shape)
# print(a.unsqueeze(-4).shape)
#for example
# b=torch.rand(32)
# f=torch.rand(4,32,14,14)
# b=b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
# print(b.shape)
###squeeze,挤压维度为1的shape
# print(b.squeeze().shape)#挤压所有可以挤压的维度
# print(b.squeeze(-1).shape)
# print(b.squeeze(1).shape)
#########expand/repeat
#expand:boradcasting改变理解方式（执行时速度快，节约内存）
#repeat:memory copied增加了数据
# print(b.expand(4,32,14,14).shape)
# print(b.expand(-1,32,-1,-1).shape)#-1计算dim
######repeat每个参数要重复的次数->memory toched
# print(b.repeat(4,32,1,1).shape)
# print(b.repeat(4,1,32,32).shape)


###################.t只使用与2D
# a=torch.rand(3,4)
# print(a.t())
# a=torch.rand(4,3,32,32)
# a1=a.transpose(1,3).view(4,3*32*32).view(4,3,32,32)#[bcwh]->[bwhc]->[bcwh]
# a1=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,3,32,32)#XXXXXX
# a2=a.transpose(1,3).contiguous().view(4,3*32*32).view(4,32,32,3).transpose(1,3)
# print(a1.shape)
# print(a2.shape)
# print(torch.all(torch.eq(a,a1)))
# print(torch.all(torch.eq(a,a2)))
# #############permute
# b=torch.rand(4,3,28,32)
# print(b.permute(0,2,3,1).shape)
############broadcasting自动扩展,节约内存消耗
#########key idea
#insert 1 dim ahead
#expand dims with size 1 to same size
#feature maps:[4,32,14,14]
#bias:[32,1,1]=>[1,32,1,1]=>[4,32,14,14]
########Is it broadcating-able
####match from last dim!!
#if current dim =1,expand to same
#if either has no dim ,insert one dim and expand to same
#otherwise,NOT broadcasting-able
########situation
#[1,32,1,1]---->[4,32,14,14]
#[14,14]--->[1,1,14,14]=>[4,32,14,14]

##############不符合broadcasting的条件
#[2,32,14,14]:
#[4,32,14,14]
#Dim 0 has Dim,can NOT insert and expand to same
#Dim 0 has distinct dim,NOT size1
#NOT broadcasting-able
#############how to understand this behavior
#########when it has no dim
####treat is as all own the same
####[class,student,scores]+[scores]
########when it has dim of size 1
####treat it shared by all
####[class],student ,scores]+[student 1]
####################################match from last dim 























