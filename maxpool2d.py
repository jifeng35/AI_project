import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('log_Maxpool2d')
test_sets = torchvision.datasets.CIFAR10('E:/CIFAR10_DataSet', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)
test_data = DataLoader(test_sets, 64)
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)
"""最大池化要求的是单精度浮点数据类型"""
input = torch.reshape(input, shape=(-1, 1, 5, 5))
print(input.shape)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
        """celi_mode True or False 会有不同大小的矩阵作为输出"""

    def forward(self, input):
        output = self.maxpool1(input)
        return output


net = Net()
# output = net(input)
# print(output)
count = 0
for data in test_data:
    imgs, targets = data
    output = net(imgs)
    writer.add_images('MaxPool2d_input', imgs, global_step=count)
    writer.add_images('MaxPool2d_output', output, global_step=count)
    count += 1

writer.close()
