#############statistics
#norm
#mean sum
#prod
#max,min,argmin,argmax
#k-thvalue,top-k
#########norm范数
#v.s. normalize ,e.g.batch_norm
#matrix norm v.s. vector norm
#####norm-p
import torch

# a=torch.full([8],1)
# b=a.view(2,4)
# c=a.view(2,2,2)
# print(a.norm(1))
# print(b.norm(1))
# print(c.norm(1))
#
# print(a.norm(2))
# print(b.norm(2))
# print(c.norm(2))
#
# print(b.norm(1,dim=1))
# print(b.norm(2,dim=1))
#
# print(c.norm(1,dim=0))
# print(c.norm(2,dim=0))
####
a=torch.arange(8).view(2,4).float()
# print(a.min())
# print(a.max())
# print(a.mean())
# print(a.prod())#累乘
# print(a.sum())
# print(a.argmax())#返回最大值的索引(vector)
# print(a.argmin())
# print(a.argmax(dim=1))
#########dim,keepdim
# a=torch.rand(2,4)
# print(a)
# print(a.max(dim=1))#维度2，输出数据+位置
# print(a.max(dim=1,keepdim=True))#保持dim
#####top-k or k-th
#######topk
########largest
#######kth value
# a=torch.rand(3,4)
# print(a)
# print(a.topk(3,dim=1))
# print(a.topk(3,dim=1),largest=False)#求最小的数值和索引
# print(a.kthvalue(8,dim=1))#求第8小的
#########compare
#####<,<=,!=,==,>,>=
#####torch.eq(a,b)#返回逐个元素比较的结果
#####torch.equal(a,b)#返回True或False
a=torch.rand(3,4)
print(a)











