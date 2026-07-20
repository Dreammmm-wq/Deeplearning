import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# 自定义 CSV 数据集
class CSVDataset(Dataset):
    def __init__(self, file_path):
        # 读取 CSV 文件
        self.data = pd.read_csv(file_path)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def __getitem__(self, idx):
        # 使用 .iloc 明确基于位置索引
        row = self.data.iloc[idx]
        # 将特征和标签分开
        features = torch.tensor(row.iloc[:-1].to_numpy(), dtype=torch.float32)  # 特征
        label = torch.tensor(row.iloc[-1], dtype=torch.float32)  # 标签
        return features, label

# 实例化数据集和 DataLoader
dataset = CSVDataset("runoob_pytorch_data.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 遍历 DataLoader
for features, label in dataloader:
    print("特征:", features)
    print("标签:", label)
    break