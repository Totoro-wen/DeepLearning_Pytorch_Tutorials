#####数据增强

#1.大数据
###the key to prevent overfitting

###sample more data？
###多样性

##限制的数据怎么班？
###减小网络的参数
###Regularization
###数据增强

#2.recap回忆

#3.数据增强
#Filp
#Rotate
#Randon move & Crop
#GAN


#4.Filp翻转
###Rotate旋转
###scale缩放
###crop part
import torchvision#视觉包
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.RandomHorizontalFlip(),#可能做,可能不做（随机）
                       transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(15),#-15 < 0 < 15
                       transforms.RandomRotation([90, 180, 270]),#随机从90,180,270选择
                       transforms.Resize([32, 32]),#scale
                       transforms.RandomCrop([28, 28]),#
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

###noise
#+N(0,0.001)

#5.数据增强将会有帮助
###但是不会太多



































