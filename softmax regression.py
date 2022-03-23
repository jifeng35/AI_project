import torch
from IPython import display
from d2l import torch as d2l

batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

