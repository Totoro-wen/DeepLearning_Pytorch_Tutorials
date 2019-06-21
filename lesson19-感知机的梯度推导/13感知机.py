#####感知机
#y=wx+b
#y=(求和)E xi*wi + b

###perceptron
#sigmiod
import torch
from torch.nn import functional as F

############单层感知机############
x=torch.randn(1,10)#1个节点10个feature
w=torch.randn(1,10,requires_grad=True)

# o=torch.sigmoid(x@w.t())
# print(o.shape)
#
# loss=F.mse_loss(torch.ones(1,1), o)
# print(loss.shape)
# loss.backward()
# print(w.grad)

############多层感知机###############
x=torch.randn(1,10)#1个节点10个feature
w=torch.randn(2,10,requires_grad=True)
o=torch.sigmoid(x@w.t())
print(o.shape)
# loss=F.mse_loss(torch.ones(1,1),o)#符合boradcasting，[1,1]->[1,2]
loss=F.mse_loss(torch.ones(1,2),o)
print(loss)
loss.backward()
print(w.grad)#w.grad是w的梯度和w是一样的shape，w'=w-lr * w.grad






