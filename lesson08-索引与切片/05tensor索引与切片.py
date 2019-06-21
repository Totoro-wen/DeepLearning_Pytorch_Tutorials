#indexing

import torch


########dim 0 frist
a=torch.rand(4,3,28,28)#batch size ,channels,h,w
# print(a[0].shape)#第一个维度torch.Size([3, 28, 28])
# print(a[0,0].shape)#第零张图片的第零个通道torch.Size([28, 28])
# print(a[0,0,2,4])#第0张图片的第0个通道上的2行4列的像素点标量tensor(0.7030)
######
# print(a[:2].shape)
# print(a[:2,:1,:,:].shape)#torch.Size([2, 1, 28, 28])
# #-1表示
# print(a[:2,1:,:,:].shape)
# print(a[:2,-1:,:,:].shape)
#######
#:  -> all
#::隔行
#:x -> 表示到x（不包含）
#n：-> n个索引到最末尾
#[start：end）
# :: -> 0:28:1(stride) -> ::1
# print(a[:,:,0:28:2,0:28:2].shape)
# a.index_select(0,[0,2])#[0,2]必须为tensor
# print(a[...].shape)#::::->a
# print(a[0,...].shape)#
# print(a[:,1,...].shape)
# print(a[...,:2].shape)
############slect by mask默认讲数据打平
print(a)
mask=a.ge(0.5)
print(torch.masked_select(a,mask))

############by flatten index
print(torch.take(a,torch.tensor([0,2,1])))
