###
#1.outline
###pooling下采样将feature map变小
###upsample上采样，与图片放大相似
###ReLU

#2.pooling
####Max pooling
#########max(1,1,5,6)=6
####Avg pooling
#########Avg(1,1,5,6)=3

#3.AleNet
#没有采用pooling，subsampling隔行采样

import torch
import torch.nn as nn
import torch.functional as F
out=torch.randn(16,7,7)
x=out
layer=nn.maxPool2d(2,stride)

#4.upsample
#1.0===F.interpolate
out=F.interpolate(x, scale_factor=2,mode='nearest')
























