import torch
import matplotlib.pyplot as plt
import random
from d2l import torch as d2l


true_w = torch.tensor([2, -2.4])  # w的真值
true_bias = 4.2  # bias偏移量的真值


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)  # 添加噪音
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 打乱list中的元素
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[indices ], labels[indices]


batch_size = 10  # k折每批数据的大小


def linreg(X, w, b):
    return torch.matmul(X, w) + b  # 矩阵乘法


def squared_loss(y_hat, y):
    """损失函数"""
    return (y_hat - y.reshape(y_hat.shape))**2/2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss


def procedure():
    features, labels = synthetic_data(true_w, true_bias, 1000)
    print('feature:', features[0], '\nlabels:', labels[0])
    d2l.set_figsize()
    # plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    plt.plot(features[:, 1].detach().numpy(), labels.detach().numpy(), marker='.', linestyle='', markersize=2,
             markerfacecolor="lime", markeredgecolor="lime")
    ax1 = plt.gca()  # 取出轴变量
    ax1.set_xlabel("features")
    ax1.set_ylabel("labels")  # 支持Letax
    ax1.set_title("Here is a Title", fontname='Arial', fontsize=10, weight='bold', style='italic')
    ax1.set_xticks([-2, -1, 0, 1, 2, 3])
    #  ax1.set_xticklabels(['J', '1', 'F', 'e', 'n', 'g']) 设置标签的刻度名称
    ax1.tick_params(axis='both', direction='in', colors='blue', length=4, width=2)
    plt.show()
    print(type(features), "\n", type(labels))
    for X, y in data_iter(batch_size, features, labels):
        print(X, f'\n{y}')
        break
    """打印数据迭代器返回的数据"""
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 根据现在的w和b计算损失函数
            l.sum().backward()  # 反向传播
            sgd([w, b], lr, batch_size)  # 小批量随机梯度下降
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch{epoch+1}, loss is {float(train_l.mean()):f}')


if __name__ == '__main__':
    procedure()

