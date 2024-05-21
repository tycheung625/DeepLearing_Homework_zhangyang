import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv("beijing2.csv")
temperature_data = data[['最高温度', '最低温度']].values.astype(float)
dates = pd.to_datetime(data['date']).values.astype(float)

# 数据标准化
scaler = MinMaxScaler()
temperature_data_normalized = scaler.fit_transform(temperature_data)

# 合并温度数据和日期数据
input_data = np.column_stack((temperature_data_normalized, dates))

# 划分训练集和测试集
train_size = int(len(input_data) * 0.8)
test_size = len(input_data) - train_size
train_data, test_data = input_data[0:train_size], input_data[train_size:len(input_data)]

# 将数据转换为PyTorch张量
train_data = torch.FloatTensor(train_data).view(-1, 1, 3)
test_data = torch.FloatTensor(test_data).view(-1, 1, 3)

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

# 实例化模型并定义损失函数和优化器
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(train_data)
    loss = criterion(y_pred, train_data[:, :, :2])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# 测试模型
model.eval()
with torch.no_grad():
    test_predictions = model(test_data)

# 反标准化预测值
test_predictions = scaler.inverse_transform(test_predictions[:, :2].numpy())

# 反标准化真实值
true_values = scaler.inverse_transform(test_data[:, :, :2].numpy().reshape(-1, 2))

# 打印预测结果和真实值
print("预测结果 vs 真实值：")
for pred, true in zip(test_predictions, true_values):
    print(f"预测值 - 最高温度: {pred[0]}, 最低温度: {pred[1]}, 真实值 - 最高温度: {true[0]}, 最低温度: {true[1]}")
