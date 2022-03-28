import torch
from torch import nn


class First_Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


first_network = First_Network()
x = torch.tensor(1.0)
output = first_network(x)
print(output)
