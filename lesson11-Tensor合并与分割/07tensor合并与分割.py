#######基础知识
#######merge or split
# cat
# stack
# split
# chunk
#####################

import torch

####1.cat
a=torch.rand(4,32,8)
b=torch.rand(5,32,8)

c=torch.cat([a,b],dim=0)#在dim=0上进行cat,允许cat的维度上不一致
# print(c)
# print(c.shape)
####2.stack
##creat new dim 在Dim前插入一个新的维度，SHAPE必须一致
a1=torch.rand(4,3,16,32)
a2=torch.rand(4,3,16,32)
c=torch.stack([a1,a2],dim=2)
print(c.shape)#torch.Size([4, 3, 2, 16, 32])
####3.cat vs. stack
#[30,28]
#[32,28]
####4.split :by len

##b,根据len拆分
# c=torch.rand(2,32,8)
# aa,bb=c.split([1,1],dim=0)
# print(aa.shape)
# print(bb.shape)
# aa,bb=c.split(1,dim=0)
# print(aa.shape)
# print(bb.shape)
# aa,bb=c.split(2,dim=0)#将C分成N快Dim=0，每一块的长度是2
# print(aa.shape)
# print(bb.shape)
####5.Chunk：by num
aa,bb=c.chunck(2,dim=0)





































