import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
data = pd.read_csv("beijing2.csv")

# 处理缺失值，这里简单地用均值填充
data = data.fillna(data.mean())
datas=[datetime.datetime.strptime(date,'%Y-%m-%d') for date in data["date"].values]
for i in range(len(data["date"].values)):
	data["date"].values[i]=datas[i]
print(data["date"].values)

# 提取特征和标签
X = data[["date", "最高温度", "最低温度"]].values # 特征为日期、最高温度和最低温度
y = data[["空气质量"]].values # 标签为空气质量
print(X.shape)
# 对特征进行归一化处理
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据转换为张量
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()

# 定义一个神经网络模型，这里使用一个简单的多层感知机（MLP）
class MLP(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) # 第一层全连接层
		self.relu = nn.ReLU() # 激活函数
		self.fc2 = nn.Linear(hidden_size, output_size) # 第二层全连接层
	def forward(self, x):
		x = self.fc1(x) # 输入经过第一层全连接层
		x = self.relu(x) # 经过激活函数
		x = self.fc2(x) # 经过第二层全连接层
		return x

# 创建模型实例，输入大小为3，隐藏层大小为10，输出大小为1
model = MLP(3, 10, 1)

# 定义优化器和损失函数，这里使用随机梯度下降（SGD）和均方误差（MSE）
optimizer = optim.SGD(model.parameters(), lr=0.01) # 学习率为0.01
criterion = nn.MSELoss() # 均方误差作为损失函数

# 定义训练的轮数（epoch）
epochs = 100

# 训练循环
for epoch in range(epochs):
# 前向传播，得到预测结果
	y_pred = model(X_train)
# 计算损失
	loss = criterion(y_pred, y_train)
# 反向传播，计算梯度
	loss.backward()
# 更新权重和偏置
	optimizer.step()
# 清零梯度
	optimizer.zero_grad()
# 打印训练信息
	print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 评估模型在测试集上的性能
with torch.no_grad(): # 不计算梯度
	y_pred = model(X_test) # 得到预测结果
	test_loss = criterion(y_pred, y_test) # 计算测试损失
	print(f"Test Loss: {test_loss.item():.4f}") # 打印测试损失

# 输出预测结果和真实结果的对比
print("Predicted values:", y_pred.numpy().flatten())
print("Actual values:", y_test.numpy().flatten())

