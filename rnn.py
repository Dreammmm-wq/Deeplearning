import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out
    

# 生成一些随机序列数据
num_samples = 1000
seq_len = 10
input_size = 5
output_size = 2  # 假设二分类问题

# 随机生成输入数据 (batch_size, seq_len, input_size)
X = torch.randn(num_samples, seq_len, input_size)
# 随机生成目标标签 (batch_size,)
Y = torch.randint(0, output_size, (num_samples,))

# ========== 打印1：整体数据集张量形状 ==========
print("="*50)
print(f"全部输入X整体形状：{X.shape}  [样本数, 序列长度, 单时刻特征数]")
print(f"全部标签Y整体形状：{Y.shape}  [样本数,]")
print("="*50)

# ========== 打印2：打印第一条样本（单条时序数据）==========
print("\n【第一条样本数据：X[0]】一条长度为10、每个时刻5个特征的序列：")
print(X[0])
print(f"单条样本形状 X[0].shape = {X[0].shape}")
print(f"第一条样本对应的真实标签 Y[0] = {Y[0].item()}")
print("-"*50)

# ========== 打印3：打印前10个样本的标签，看标签分布 ==========
print("\n前10个样本真实标签列表：")
print(Y[:10])
print("="*50)

# 创建数据加载器
dataset = TensorDataset(X, Y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 打印4：取出一个batch，看dataloader真实输出格式 ==========
print("\n===== 读取一个batch的数据查看 =====")
# 迭代器取第一批数据
one_batch_inputs, one_batch_labels = next(iter(train_loader))
print(f"一个batch输入形状：{one_batch_inputs.shape} [batch=32, seq_len=10, input=5]")
print(f"一个batch标签形状：{one_batch_labels.shape}")
print("\nbatch内前3条序列：")
print(one_batch_inputs[:3])
print("\nbatch内前10个真实标签：")
print(one_batch_labels[:10])
print("="*60)

# 模型实例化
model = SimpleRNN(input_size=input_size, hidden_size=64, output_size=output_size)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # ========== 可选：每个epoch第一个batch打印真实标签对比 ==========
        if batch_idx == 0 and epoch == 0:
            print(f"\n【第{epoch+1}轮，第一个batch真实标签】")
            print(labels)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    total = 0
    correct = 0
    # ========== 测试阶段打印一组真实标签与预测值对比 ==========
    print("\n===== 测试集：真实标签 vs 模型预测结果 =====")
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 只打印第一个batch做示例
        if batch_idx == 0:
            print("该批次真实标签：", labels)
            print("模型预测类别：", predicted)
            print("预测正确标记(1正确/0错误)：", (predicted == labels).int())
            break

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")