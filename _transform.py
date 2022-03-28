from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs_tensor")

"""ToTemsor的使用"""
"""实例化一个ToTensor类,将图片放入,则图片变为Tensor数据类型"""
img_path = 'test_data/train/bees_image/16838648_415acd9e3f.jpg'
img = Image.open(img_path)
tensor_convert = transforms.ToTensor()
img_tensor = tensor_convert(img)

writer.add_image("Tensor_img", img_tensor, global_step=1)

"""Normalize的使用"""
print(img_tensor[0][0][0])
normal_convert = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normal = normal_convert(img_tensor)
print(img_normal[0][0][0])

writer.add_image("Normalize_img", img_normal, global_step=1)

"""Resize的使用"""
resize_convert = transforms.Resize((1080, 1920))
img_resize = resize_convert(img)
print(type(img_resize))  # 此处的type为 <class 'PIL.Image.Image'>
"""需要进行类型转换才能在tensorboard中输出结果"""
img_resize_show = tensor_convert(img_resize)

writer.add_image("Resize_img", img_resize_show, global_step=1)

"""Compose类可以讲任意变换组合在一起"""
trans_compose = transforms.Compose([resize_convert, tensor_convert, normal_convert])
"""compose中的转换器有序且上一个的输入与下一个的输出格式必须一致"""
img_compose = trans_compose(img)

writer.add_image("Compose_img", img_compose, global_step=1)

"""RandomCrop随机裁剪"""
trans_crop = transforms.RandomCrop(50)
trans_compose = transforms.Compose([trans_crop, tensor_convert])
for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image('RandomCrop_img', img_crop, global_step=i + 1)

writer.close()
# print(img_tensor)
