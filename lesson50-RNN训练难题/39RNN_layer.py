###


###1.input dim,hidden dim
import torch
import torch.nn as nn

# rnn = nn.RNN(100, 10)
# print(rnn._parameters.keys())
# #######################Win,   Wnn
# print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)
# #######################Bin,   Bnn
# print(rnn.bias_hh_l0.shape, rnn.bias_ih_l0.shape)

###2.nn.RNN
#####__init__
#input_size
#hidden_size
#num_layer

#####out,ht=forward(x,h0)
#x:[seq len , b ,word vec]
#h0/ht:[num layers, b, h dim](ht每个时刻的返回)
#out:[seq len, b,h dim]

###3.单个RNN
# rnn = nn.RNN(input_size=100, hidden_size=20,num_layers=1)
# print(rnn)
# x=torch.randn(10,3,100)
# ######################h0,一层，3个句子，shape
# out,h=rnn(x,torch.zeros(1,3,20))
#
# print(out.shape, h.shape)

###4.多层

######h:最后一个时间戳上的所有memory状态
######out：最有时间戳
# rnn = nn.RNN(input_size=100, hidden_size=10,num_layers=2)
# print(rnn)
# print(rnn._parameters.keys())
# #######################Win,   Wnn
# print(rnn.weight_hh_l0.shape, rnn.weight_ih_l0.shape)
# print(rnn.weight_hh_l1.shape, rnn.weight_ih_l1.shape)
# #######################Bin,   Bnn
# print(rnn.bias_hh_l0.shape, rnn.bias_ih_l0.shape)


###5.[T,b,h_dim],[layers,b,h_dim]
rnn = nn.RNN(input_size=100, hidden_size=20,num_layers=4)

print(rnn)
x=torch.randn(10,3,100)
out,h=rnn(x)
print(out.shape, h.shape)

###6.nn.RNNCell不会循环T次
#####__init__
#



#ht=rnncell(xt,ht_1)
##xt：[b, word vec]
##ht_1/ht:[num layers, b, h dim]
##out = torch.stack([h1,h2,...,ht])

###7.functional
# cell1=nn.RNNCell(100, 30)
# cell2=nn.RNNCell(30, 20)
#
# h1= torch.zeros(3, 30)
# h2= torch.zeros(3, 20)
#
# for xt in x:
#     h1=cell1(xt,h1)#添加循环
#     h2=cell2(h1,h2)
# print(h2.shape)


