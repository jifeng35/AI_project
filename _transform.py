from PIL import Image
from torchvision import transforms

img_path = 'test_data/train/bees_image/16838648_415acd9e3f.jpg'
img = Image.open(img_path)

"""实例化一个ToTensor类,将图片放入,则图片变为Tensor数据类型"""
tensor_convert = transforms.ToTensor()
img_tensor = tensor_convert(img)

print(img_tensor)
