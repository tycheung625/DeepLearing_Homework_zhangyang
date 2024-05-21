import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


data = pd.read_csv('beijing2.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')


# 选取最高温度和最低温度作为预测目标
target_cols = ['最高温度', '最低温度']
data = data[target_cols]

# 数据预处理
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 划分数据集为训练集和测试集
train_size = int(len(data) * 0.8)
train_data, test_data = data_scaled[0:train_size], data_scaled[train_size:]

# 将数据转换为模型可用的序列数据

def create_sequences(data, seq_length):
    sequences = []
    targets = []
    dates = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        target = data[i + seq_length:i + seq_length + 1].flatten()
        date_index = i + seq_length

        date = data[date_index,0]
        sequences.append(seq)
        targets.append(target)
        dates.append(date)
    return torch.tensor(sequences), torch.tensor(targets), dates
    
seq_length = 15

X_train, y_train, train_dates = create_sequences(train_data, seq_length)
X_test, y_test, test_dates = create_sequences(test_data, seq_length)



# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 定义模型参数
input_size = len(target_cols)
hidden_size = 128
output_size = len(target_cols)
num_layers = 3


model = RNN(input_size, hidden_size, output_size, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 将数据移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
model.to(device)

# 训练模型
num_epochs = 300
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train.float())
    optimizer.zero_grad()
    loss = criterion(outputs, y_train.float())
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    test_outputs = model(X_test.float())
    mse = criterion(test_outputs, y_test.float())
    print(f'Mean Squared Error on Test Data: {mse.item():.4f}')

test_dates_original = data.index[train_size + seq_length:]

# 将预测结果逆标准化
test_outputs = scaler.inverse_transform(test_outputs.cpu().numpy())
y_test = scaler.inverse_transform(y_test.cpu().numpy())

# 打印部分预测结果
for i in range(10):
    formatted_date = test_dates_original[i].strftime('%Y-%m-%d')
    print(f'Date: {formatted_date}, Predicted: {test_outputs[i]}, Actual: {y_test[i]}')


# 绘制最高温度和最低温度拟合曲线
plt.figure(figsize=(12, 6))

# 绘制最高温度拟合曲线
plt.plot(test_dates_original, y_test[:, 0], label='Actual High Temperature', marker='o', color='blue')
plt.plot(test_dates_original, test_outputs[:, 0], label='Predicted High Temperature', marker='o', linestyle='dashed', color='orange')

# 绘制最低温度拟合曲线
plt.plot(test_dates_original, y_test[:, 1], label='Actual Low Temperature', marker='o', color='green')
plt.plot(test_dates_original, test_outputs[:, 1], label='Predicted Low Temperature', marker='o', linestyle='dashed', color='red')

plt.xlabel('Date')
plt.ylabel('Temperature')
plt.title('Temperature Prediction')
plt.legend()
plt.show()
