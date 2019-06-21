###visdom可视化
#1.TensorBoard？google


#2.pip install tensorboardX(pytorch)
# from tensorboardX import SummaryWriter
#
# writer = SummaryWriter()
# writer.add_scalar('data/scalar1',dummy_s1[0],n_iter)
#
# writer.add_scalar('data/scalar_group',{'xsinx': n_iter*np.sin(n_iter),
#                                        'xcosx': n_iter*np.cos(n_iter),
#                                        'arctanx': np.arctan(n_iter)},n_iter)
#
# writer.add_image('Image',x, n_iter)
# writer.add_text('Text','text logged at step:' + str(n_iter), n_iter)
#
# for name ,param in resnet18.named_paraments():
#     writer.add_histogram(name,param.clone().cpu().data.numpy(), n_iter)
#
# writer.close()

#最大的问题：tensorboard抽取的是numpy，跟tensor做match，先转换到cpu，在转换到numpy


#3.visdom from facebook(接收tensor，效率高，tensorboard会将数据写到文件，占用大量资料)
##同时监听多个，1s刷新一次
##pip install visdom
##python -m visdom.server
from visdom import Visdom

####单个线条
#实例化
# viz=Visdom()
# #创建一个窗口trian loss
# #           Y,  x ,     ID(defult窗口=main),其他参数,窗口名称
# viz.line([0.],[0.],win='train_loss',opts=dict(title='train loss'))
# #         直接数据       x坐标代表时间戳                      添加方式
# viz.line([loss.item()], [global_step], win ='train_loss', update='append')

########lines：multi-traces#########
viz=Visdom()
#创建一个窗口trian loss
#           Y,  x ,     ID(defult窗口=main),其他参数,legend,y1和y2的
viz.line([[0.0, 0.0]],[0.],win='test',opts=dict(title='test loss&acc',
                                                legend=['loss','acc.']))
#         直接数据       x坐标代表时间戳                      添加方式
viz.line([[test_loss, correct / len(test_loader.dataset)]],
         [global_step], win ='test', update='append')

########visual X####################
viz=Visdom()
viz.images(data.view(-1,1,28,28),win='x')
viz.test(str(pred.detach().cpu().numpy()),win='pred',
         opts=dict(title='pred')










