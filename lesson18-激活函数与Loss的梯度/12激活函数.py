######激活函数与loss的梯度

#1.sigmoid/logistic
##f(x)=1/(1+e^(-x))
##缺陷：在负无穷正无穷时,梯度趋近于0,造成梯度弥散,长时间得不到更新
import torch

# a=torch.linspace(-100,100,10)
# print(a)
# print(torch.sigmoid(a))
#2.tanh(RNN用的比较多)
##f(x)=tanh(x)=(e^x-e^(-x))/(e^x+e^(-x))=2sigmoid(2x)-1


#3.Rectified Liner Unit(ReLU)
##f(x)=0,x<0
##    =x,x>0
from torch.nn import functional as F
# a=torch.linspace(-1,1,10)
# print(a)
# print(torch.relu(a))
# print(F.relu(a))



#4.Typical Loss
###Mean Squared Loss均方差
###Cross Entropy Loss分类误差
#####binary
#####multi-class
#####+softmax
#####Leave it to Logistic Regression Part

# x=torch.ones(1)
# w=torch.full([1],2)
# mse=F.mse_loss(torch.ones(1),x*w)
# print(mse)
##分类算法：SVM,决策树etc有监督
##聚类算法：层次，网格，划分（kmeans对离群点的效果差，采用的欧式距离），密度无监督

#静态网址：HTML+css
#动态网址：jbs

#########autograd.grad求导############
# print(torch.autograd.grad(mse,[w]))#pred,[w1,w2,w3...]求导
# w.requires_grad_()#需要梯度信息  ==  w=torch.tensor([1],requires_grad=True)
# mse=F.mse_loss(torch.ones(1),x*w)#更新图
# print(w)
# c=torch.autograd.grad(mse,[w])#loss对w的偏导
# print(c)
##############loss.backward求导##########
# x=torch.ones(1)
# w=torch.full([1],2)
# w.requires_grad_()
# torch.autograd.grad(mse,[w])
# mse=F.mse_loss(torch.ones(1),x*w)#动态图见图
# mse.backward()#向前传播
# print(w.grad)#打印w的导数


####################softmax###############
#soft version of max
#######probabilties
#######大的值更大，小的更小，压缩到【0,1】
#derivative
a=torch.rand(3)
print(a)
print(type(a))
a.requires_grad_()
print(a)
p=F.softmax(a,dim=0)#可能a=[batch_size,feature]
# p.backward(retain_graph=True)#完成一次反向传播，把梯度写到w.grad，将图的梯度信息清除,标志只保留一次
#？RuntimeError: grad can be implicitly created only for scalar outputs
# p.backward(retain_graph=True)#梯度信息不会被清除
# p.backward()
p=F.softmax(a,dim=0)
c=torch.autograd.grad(p[1],[a],retain_graph=True)#p1对ai求导，i属于[0,2]
print(c)
d=torch.autograd.grad(p[2],[a])#i=j时，梯度为正数
print(d)





