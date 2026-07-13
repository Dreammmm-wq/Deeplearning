import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 定义数据预处理的流水线
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将图像调整为 128x128
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载图像
image = Image.open('D:\Program\Python\DeepLearning\image\OIP-C.webp')  # 替换为你的图像路径



# 应用预处理
image_tensor = transform(image)
print(image_tensor.shape)  # 输出张量的形状

