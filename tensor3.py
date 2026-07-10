import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print("Device:", x.device)  # 设备
print("CUDA 可用性:", torch.cuda.is_available())  # 返回 True 或 False