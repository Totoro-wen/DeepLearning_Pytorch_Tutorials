import torch
import os , glob
import random, csv
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from PIL import Image

class Pokemon(Dataset):

    def __init__(self, root, resize, mode):
        super(Pokemon, self).__init__()

        #保存参数
        self.root = root
        self.resize = resize

        #编码种子,建立表格映射关系
        self.name2label = {} # "sq...".0
        for name in sorted(os.listdir(os.path.join(root))):

            #listdir()返回的顺序不固定+排序sorted()
            if not os.path.isdir(os.path.join(root, name)):
                continue

            #将当前最新的长度作为label值
            self.name2label[name] = len(self.name2label.keys())

        # print(self.name2label)#{'mewtwo': 2, 'bulbasaur': 0, 'squirtle': 4, 'charmander': 1, 'pikachu': 3}

        #image, label
        #这里保存所有的image_path而不是image本身,因为如果一次把所有图片加载景来可能爆内存

        self.images, self.labels = self.load_csv('images.csv')


        if mode == 'training': #60%
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]

        elif mode == 'val': #20%
            self.images = self.images[int(0.6 * len(self.images)):int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8 * len(self.labels))]

        else: #20%
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]


    def load_csv(self, filename):

        #如果不存在csv文件则创建
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                #Pokemon\\mewtwo\\0001.jpg
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            #1165, 'Pokemon/mewtwo/00000151.jpg'
            print(len(images), images)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode = 'w', newline = '') as f:
                writer = csv.writer(f)
                for img in images:#Pokemon\\mewtwo\\0001.jpg(windows)
                    name = img.split(os.sep)[-2]#os.sep跟平台无关
                    label = self.name2label[name]
                    # Pokemon\\mewtwo\\0001.jpg, 1
                    writer.writerow([img, label])
                print('writen csv file:', filename)


        #reader from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv .reader(f)
            for row in reader:
                # Pokemon\\mewtwo\\0001.jpg, 1
                img, label = row
                label = int(label)

                images.append(img)
                labels.append(label)

        # 保证数据和长度是一样的
        assert len(images) == len(labels)

        return images, labels

    def __len__(self):

        # 裁剪后的
        return len(self.images)

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x_hot = (x - mean)/std # ==normalize()
        # x = x_hat*std + mean
        # x: [c, h, w]
        # mean:[3] =>[3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)## unsqueeze(1)在后面插入一个维度
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat * std + mean

        return x

    def __getitem__(self, idx):
        # idx=[0,len(iamges)]
        # self.images, self.labels
        # img: 'Pokemon/mewtwo/00000151.jpg'
        # label; 0
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),  # string path = > image data
            transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),#为后面的旋转做好准备放大
            transforms.RandomRotation(15),#rotate比较大的话,可能造成网络不收敛的情况
            transforms.CenterCrop(self.resize),#去除旋转产生的黑边
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),#统计得到的数据imagenet上的均值和方差,服从[-1,1]之间的分布,而visdom接受的是[0,1]

        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label
    
# 验证时使用
def main():

    import visdom
    # pip install visdom
    # 在终端下,python -m visdom.server
    # 再打开浏览器
    import time
    import torchvision
    viz = visdom.Visdom()

    #####################另外一种简单的数据集载入方法############################
    #使用情况:图片规整
    # tf1 = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))#数据有影响,显示需要denormalize
    #
    # ])
    # db = torchvision.datasets.ImageFolder(root='Pokemon', transform=tf1)#按照二级目录的存储方式加载
    #
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)#打印编码方式即标签对应值
    #
    # for x, y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)

    ######################常用数据集载入方法###################################
    db = Pokemon('Pokemon', 224,  'train')

    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, y)

    # viz.image(x, win='sample_x', opts=dict(title='sample_x'))
    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    
    # 训练时,加载一个batch
    loader = DataLoader(db, batch_size=32, shuffle=True, num_workers=8)#8个线程

    for x, y in loader:
        viz.images(db.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))

        time.sleep(10)


if __name__ == "__main__":
    main()














