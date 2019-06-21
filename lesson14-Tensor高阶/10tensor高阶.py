#tensor advanced operation
#####where
#####gather
import torch

####torch.where(condition,x,y)->tensor#out=x,if cond; out = y,otherwise
######for,if在cpu上运行


####torch.gather(input,dim,index,out=None)#查表的过程==收集


####retrieve global label
######argmax(pred) to get relative labeling
######on some ,our label is distinct from relative labeling


prob=torch.rand(4,10)
idx=prob.topk(dim=1,k=3)
idx=idx[1]
label=torch.arange(10)+100
print(label)
a=torch.gather(label.expand(4,10),dim=1,index=idx.long())
print(a)

