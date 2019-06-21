#######
#Himmelblau function检测优化器
#######等待运行#########
import torch
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def himmelblau(x):
    return (x[0] ** 2+x[1]-11)**2 + (x[0]+x[1]**2-7)**2

x=np.arange(-6,6,0.1)
y=np.arange(-6,6,0.1)
print("x,y range:",x.shape,y.shape)
X,Y=np.meshgrid(x,y)#将xy坐标拼接在一起
print("X,Y maps:",X.shape,Y.shape)
Z=himmelblau([X,Y])


fig=plt.figure('himelblau')
ax=fig.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

#############################
x=torch.tensor([0.,0.],requires_grad=True)#初始化,初始化关键
optimizer=torch.optim.Adam([x],lr=1e-3)#x'=x-(1e-3)*x.grad(x的梯度)，y'=y-(1e-3)*y.grad
for step in range(20000):
    pred=himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()#更新x'和y'
    if step %2000 ==0:
        print('stepp {}:x={},f(x)={}'
              .format(step,x.tolist(),pred.item()))



















