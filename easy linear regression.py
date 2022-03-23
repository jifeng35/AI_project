import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -2.4])
true_bias = 4.2
features, labels = d2l.synthetic_data(true_w, true_bias, 1000)
batch_size = 10


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


data_iter = load_array((features, labels), batch_size)


def procedure():
    print(next(iter(data_iter)))
    net = nn.Sequential(nn.Linear(2, 1))  # 定义训练的网络
    net[0].weight.data.normal_(0, 0.01)  # normal_(mean=0, std=1, , gengerator=None*)
    # 将tensor用均值为 mean 和标准差为 std 的正态分布填充。
    net[0].bias.data.fill_(0)  # 将tensor用指定数值填充
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()  # 进行单词优化
        l = loss(net(features), labels)
        print(f'epoch{epoch+1},loss{l:f}')


if __name__ == '__main__':
    procedure()
