import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('E:/CIFAR10_DataSet', train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        return self.conv1(x)


writer = SummaryWriter('log_conv2d')
Net = net()
print(Net)
count = 0
for data in dataloader:
    imgs, targets = data
    output = Net(imgs)
    # torch.Size([64, 3, 32, 32]) (batch_size, channels, Hegiht, Width)
    writer.add_images('original imgs', imgs, global_step=count)
    # torch.Size([64, 6, 30, 30]) 此处输出为6通道 无法显示
    output = torch.reshape(output, [-1, 3, 30, 30])
    """-1写入,程序会自动计算合适的数值,进行reshape"""
    writer.add_images('result imgs', output, global_step=count)
    count += 1
    # print(imgs.shape)
    # print(output.shape)

writer.close()
