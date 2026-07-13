import torch
import numpy as np
import matplotlib.pyplot as plt

# 随机种子，确保每次运行结果一致
torch.manual_seed(42)

# 生成训练数据
X = torch.randn(100, 2)  # 100 个样本，每个样本 2 个特征
true_w = torch.tensor([2.0, 3.0])  # 假设真实权重
true_b = 4.0  # 偏置项
Y = X @ true_w + true_b + torch.randn(100) * 0.1  # 加入一些噪声

# 打印部分数据
print(X[:5])
print(Y[:5])

import torch.nn as nn

# 定义线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 定义一个线性层，输入为2个特征，输出为1个预测值
        self.linear = nn.Linear(2, 1)  # 输入维度2，输出维度1

    def forward(self, x):
        return self.linear(x)  # 前向传播，返回预测结果

# 创建模型实例
model = LinearRegressionModel()

# 损失函数（均方误差）
criterion = nn.MSELoss()

# 优化器（使用 SGD 或 Adam）
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 学习率设置为0.01

# 也可以使用 Adam 优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for i in range(1000):  # 训练1000次
    model.train()  # 设置模型为训练模式
    optimizer.zero_grad()  # 清空梯度
    outputs = model(X)  # 前向传播，得到预测值
    loss = criterion(outputs.squeeze(), Y)  # 计算损失
    loss.backward()  # 反向传播，计算梯度
    optimizer.step()  # 更新参数

    if (i + 1) % 100 == 0:  # 每100次打印一次损失
        print(f'Epoch [{i + 1}/1000], Loss: {loss.item():.4f}')