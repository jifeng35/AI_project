# 这是一个示例 Python 脚本。
import torch


# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
def m_grad():
    x = torch.arange(4.0)
    x.requires_grad_(True)
    print(x)

    y = 2 * torch.dot(x, x)
    y.requires_grad_(True)
    print(y)
    y.backward()
    print(x.grad)

    print(4 * x == x.grad)

    x.grad.zero_()  # 清零之前的梯度, 否则pytorch默认梯度累加
    y = x.sum()
    y.backward()
    print(x.grad)

    x.grad.zero_()
    y = x * x
    y.sum().backward()  # todo 求导需要标量 x^2 求导等于 2x
    print(x.grad)

    x.grad.zero_()
    y = x * x
    u = y.detach()  # todo 将变量从当前的计算图中分离出来,数值不发生改变
    # todo (即与x无关.求导得出的结果为常数)
    z = u * x
    z.sum().backward()  # z的求导可证明, u与x无关
    print(x.grad)

    x.grad.zero_()
    y.sum().backward()  # y的导数与x有关
    print(2 * x == x.grad)


def f(a):
    """无实际意义,仅用作有效性证明"""
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    m_grad()
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()
    print(a.grad == d / a)  # 无论如何调用循环,判断等控制语句,都能计算出其梯度

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
