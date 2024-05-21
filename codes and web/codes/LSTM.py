import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# 训练集
train_data = pd.read_csv('beijing2.csv')
train_dates = pd.to_datetime(train_data['date'])  # 转换日期格式
train_temperature_max = train_data['最高温度']
train_temperature_min = train_data['最低温度']

# 测试集
test_data = pd.read_csv('test.csv')
test_dates = pd.to_datetime(test_data['date'])  # 转换日期格式
test_temperature_max = test_data['最高温度']
test_temperature_min = test_data['最低温度']


scaler_temp = MinMaxScaler()
scaler_date = MinMaxScaler()


train_temperature_max_scaled = scaler_temp.fit_transform(np.array(train_temperature_max).reshape(-1, 1))
train_temperature_min_scaled = scaler_temp.fit_transform(np.array(train_temperature_min).reshape(-1, 1))
train_dates_scaled = scaler_date.fit_transform(np.array(train_dates).reshape(-1, 1).astype('float32'))  # 将日期转换为数值


X_train = torch.tensor(
    np.column_stack((train_dates_scaled, train_temperature_max_scaled, train_temperature_min_scaled)),
    dtype=torch.float32)
y_train = torch.tensor(np.column_stack((train_temperature_max_scaled, train_temperature_min_scaled)),
                       dtype=torch.float32)


test_temperature_max_scaled = scaler_temp.transform(np.array(test_temperature_max).reshape(-1, 1))
test_temperature_min_scaled = scaler_temp.transform(np.array(test_temperature_min).reshape(-1, 1))
test_dates_scaled = scaler_date.transform(np.array(test_dates).reshape(-1, 1).astype('float32'))  # 将日期转换为数值


X_test = torch.tensor(np.column_stack((test_dates_scaled, test_temperature_max_scaled, test_temperature_min_scaled)),
                      dtype=torch.float32)
y_test = torch.tensor(np.column_stack((test_temperature_max_scaled, test_temperature_min_scaled)), dtype=torch.float32)



class WeatherLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(WeatherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])
        out = self.fc1(out)
        out = self.fc2(out)
        return out



input_size = 3  
hidden_size = 150  
output_size = 2  
num_layers = 2  
model = WeatherLSTM(input_size, hidden_size, output_size, num_layers)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 将训练集数据转换成适合LSTM的输入格式
X_train = X_train.view(X_train.shape[0], 1, X_train.shape[1])

# 训练模型
num_epochs = 150  # 增加训练轮次
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 使用测试集评估模型
model.eval()
with torch.no_grad():
    X_test = X_test.view(X_test.shape[0], 1, X_test.shape[1])
    test_outputs = model(X_test)

# 将测试集数据反标准化
test_outputs = scaler_temp.inverse_transform(test_outputs.numpy())
y_test = scaler_temp.inverse_transform(y_test.numpy())


plt.figure(figsize=(12, 6))

# 绘制最高温度拟合曲线
plt.subplot(1, 2, 1)
plt.plot(test_dates, test_outputs[:, 0], label='Predicted Max Temp')
plt.plot(test_dates, y_test[:, 0], label='Actual Max Temp', linestyle='--')
plt.title('Max Temperature Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()

# 绘制最低温度拟合曲线
plt.subplot(1, 2, 2)
plt.plot(test_dates, test_outputs[:, 1], label='Predicted Min Temp')
plt.plot(test_dates, y_test[:, 1], label='Actual Min Temp', linestyle='--')
plt.title('Min Temperature Prediction')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()

plt.tight_layout()
plt.show()
