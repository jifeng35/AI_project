import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#  准备测试集
test_data = torchvision.datasets.CIFAR10('E:/CIFAR10_DataSet', train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("log_dataloader")
step = 0
num_epoch = 2
for i in range(num_epoch):
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"test_data{i}", imgs, step)
        step += 1
        # print(targets)
        # print(imgs.shape)
        # print('\n')

writer.close()
