#-*- coding:utf-8 -*-
import  torch
from    torch import nn, optim, autograd
import  numpy as np
import  visdom
from    torch.nn import functional as F
from    matplotlib import pyplot as plt
import  random

h_dim = 400
batchsz = 512
viz = visdom.Visdom()

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        #       , 2随意 | , 2 ==x,y是为了显示
        # z: [b , 2] => [b, 2]
        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2),
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()     #压缩到[0,1]
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)

def data_generator():
    '''
    8-gaussian mixture models
    :return:
    '''
    scale = 2.
    centers = [ #8个均值点
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers]
    while True:
        dataset = []
        for i in range(batchsz):

            point = np.random.randn(2) * .02# sample一个点
            center = random.choice(centers)
            # N(0, 1) + center_x1/x2
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)

        dataset = np.array(dataset, dtype='float32')#转换为numpy
        dataset /= 1.414  # stdev放缩
        yield dataset   # 返回数据并保存状态,不会重新开始,数据生成器

    # for i in range(100000//25):
    #     for x in range(-2, 3):
    #         for y in range(-2, 3):
    #             point = np.random.randn(2).astype(np.float32) * 0.05
    #             point[0] += 2 * x
    #             point[1] += 2 * y
    #             dataset.append(point)
    #
    # dataset = np.array(dataset)
    # print('dataset:', dataset.shape)
    # viz.scatter(dataset, win='dataset', opts=dict(title='dataset', webgl=True))
    #
    # while True:
    #     np.random.shuffle(dataset)
    #
    #     for i in range(len(dataset)//batchsz):
    #         yield dataset[i*batchsz : (i+1)*batchsz]


def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    xr:真实sample出来的x
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda() # [16384, 2]
        disc_map = D(points).cpu().numpy() # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()


    # draw samples
    with torch.no_grad():
        z = torch.randn(batchsz, 2).cuda() # [b, 2]
        samples = G(z).cpu().numpy() # [b, 2]
    xr = xr.data.cpu().numpy()
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d'%epoch))


def weights_init(m):
    if isinstance(m, nn.Linear):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

def gradient_penalty(D, xr, xf):
    """

    :param D:
    :param xr:
    :param xf:
    :return:
    """
    LAMBDA = 0.3

    # only constrait for Discriminator
    xf = xf.detach()
    xr = xr.detach()

    # [b, 1] => [b, 2]
    alpha = torch.rand(batchsz, 1).cuda()
    alpha = alpha.expand_as(xr)

    interpolates = alpha * xr + ((1 - alpha) * xf)
    interpolates.requires_grad_()

    disc_interpolates = D(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gp

def main():

    # 设置种子,减少随机性
    torch.manual_seed(23)
    # numpy设置种子
    np.random.seed(23)

    G = Generator().cuda()#.cuda()===pytorch0.3; .to()===pytorch0.4
    D = Discriminator().cuda()
    G.apply(weights_init)
    D.apply(weights_init)

    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))#lr=1e-3
    optim_D = optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.9))


    data_iter = data_generator()
    print('batch:', next(data_iter).shape)

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss',
                                                 legend=['D', 'G']))#legend表示图标哦

    for epoch in range(50000):

        # 1. train discriminator for k steps
        for _ in range(5):

            #1.1 train on real data

            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()

            # [b, 2]  => [b, 1]
            predr = (D(xr))
            # max log(lossr)
            # max predr
            lossr = - (predr.mean())

            #1.2 train on fake data
            # [b, 2]
            z = torch.randn(batchsz, 2).cuda()
            # stop gradient on G
            # [b, 2]
            xf = G(z).detach()#.detach()==tf.stop_grafient(),第一,梯度不会往前传递;第二,只优化discriminator
            # [b]
            predf = (D(xf))
            # min predf
            lossf = (predf.mean())

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            loss_D = lossr + lossf + gp
            optim_D.zero_grad()
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()


        # 2. train Generator
        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        predf = (D(xf))
        # max predf
        loss_G = - (predf.mean())
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()


        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

            generate_image(D, G, xr, epoch)

            print(loss_D.item(), loss_G.item())






if __name__ == '__main__':
    main()
