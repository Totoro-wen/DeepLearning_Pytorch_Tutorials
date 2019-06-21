#####交叉熵
#1.class for classification
###MSE
###Cross Entropy Loss
###Hinge Loss(SVM比较多)
#####E max(0,1-yi * h(xi))

#2,Entropy
###Uncertainty
###Measure of surprise
###Higer entropy=less info
#####Entropy = -E p(i)*logP(i)

import torch

# a=torch.full([4],1/4)
# print(a*torch.log2(a))
# print(-(a*torch.log2(a)).sum())#entropy

#3.Cross Entropy
##H(p,q)=Ep(x)logq(x)
##H(p,q)=H(p)+Dkl(p|q)
##P=Q
####kl divergence散度，kl=0，两个分布（q,p）重合
####cross entropy =entropy
##for one-hot encoding
####entrpy=1log1=0

#4.Binary Classification
##见PPT


#5.Why not use MSE
####sigmoid + MSE
######gradient vanish
####converge slower
####But ,sometimes
######eg.meta-learing

#总结神经网络
#softmax+ cross entropy一起使用防止数据不稳定

from torch.nn import functional as F
x=torch.randn(1,784)
w=torch.randn(10,784)

logits=x@w.t()
print("logits:",logits)

pred=F.softmax(logits,dim=1)
print("pred:",pred)

pred_log=torch.log(pred)
print("pred_log:",pred_log)

out1=F.cross_entropy(logits,torch.tensor([3]))#将softmax和log,nll_loss打包在一起
print("out1:",out1)

out2=F.nll_loss(pred_log,torch.tensor([3]))
print("out2:",out2)

























