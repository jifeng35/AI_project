import torch
import torch.nn.functional as func

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
"""卷积核 kernel"""
input = torch.reshape(input, shape=(1, 1, 5, 5))
kernel = torch.reshape(kernel, shape=(1, 1, 3, 3))

output_stride1 = func.conv2d(input, kernel, stride=1)
output_stride2 = func.conv2d(input, kernel, stride=2)
output_stride1_padding1 = func.conv2d(input, kernel, stride=1, padding=1)
"""stride 卷积核每个step移动的距离"""
"""padding 在原矩阵周围填充{padding}行0 会导致整个矩阵变大,卷积结果也变大 """
print(output_stride1)
print(output_stride2)
print(output_stride1_padding1)
