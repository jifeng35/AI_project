import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = 'test_data/train/ants_image/0013035.jpg'
img = Image.open(img_path)
# img.show()
img_array = np.array(img)

writer = SummaryWriter("logs")

writer.add_image("test", img_array, global_step=2, dataformats='HWC')
writer.add_image("train", img_array, global_step=1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y = x", i, i)  # (title, y, x)

writer.close()
