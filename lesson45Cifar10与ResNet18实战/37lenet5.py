import torch
from torch import nn
from torch.nn import functional as F
class Lenet5(nn.Module):
    """
    for cifar10 dataset
    """
    def __init__(self):
        super(Lenet5, self).__init__()

        self.conv_unit=nn.Sequential(
            #x:[b, 3, 32, 32] = > [b, 6, ]
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            #

        )
        #flatten
        #fc unit
        self.fc_unit=nn.Sequential(
            # nn.Linear(2,120),#nn.Linear是一个类，需要初始化
            # nn.ReLU(),
            # nn.Linear(120,84),
            # nn.ReLU(),
            # nn.Linear(84,10)
            nn.Linear(32 * 5 * 5, 32),
            nn.ReLU(),
            # nn.Linear(120, 84),
            # nn.ReLU(),
            nn.Linear(32, 10)

        )

        #
        tmp=torch.rand(2,3,32,32)
        out=self.conv_unit(tmp)
        # [2, 16, 5, 5]
        print('con_out:',out.shape)

        #use CrossEntropyLoss 分类问题
        #回归问题：MSELoss
        # self.criteon = nn.CrossEntropyLoss()



    def forward(self, x):
        '''

        :param x:[b,3,32,32]
        :return:
        '''
        batchsz = x.size(0)
        #[b,3,32,32] = > [b,16,5,5]
        x=self.conv_unit(x)
        #[b,3,32,32] = > [b,16*5*5]
        x = x.view(batchsz, 32*5*5)#x = x.view(batchsz, -1)#-1表示推算
        #[b,16*5*5] = > [b,10]
        logits=self.fc_unit(x)#logits指

        # #[b,10]
        # pred=F.softmax(logits,dim=1)#F.softmax直接的一个函数，不需要初始化类
        # loss=self.criteon(logits, y)
        return logits

def main():

    net=Lenet5()
    #
    tmp = torch.rand(2, 3, 32, 32)
    out = net(tmp)
    # [2, 16, 5, 5]
    print('lenet_out:', out.shape)


if __name__=='__main__':
    main()







































