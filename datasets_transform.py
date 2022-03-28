import torchvision
from torch.utils.tensorboard import SummaryWriter

trans = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='../CIFAR10_DataSet', train=True, transform=trans, download=True)
test_set = torchvision.datasets.CIFAR10(root='../CIFAR10_DataSet', train=False, transform=trans, download=True)

# print(train_set[0])
# print(train_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

print(test_set[1])

writer = SummaryWriter("datasets_transform")
for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_img', img, global_step=i)
