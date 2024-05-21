import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
# 读取数据
data = pd.read_csv('beijing2.csv')

# 提取特征和标签
features = data[['date', '最高温度', '最低温度']]
temperature_labels = data[['最高温度', '最低温度']]
air_quality_labels = data['空气质量'].astype(float)  # 将空气质量列转换为浮点数

# 将日期转换为天数，作为一个简单的特征
features['date'] = features['date'].apply(lambda x: (datetime.strptime(x, "%Y/%m/%d") - datetime(1970, 1, 1)).days)

# 数据预处理
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 选择固定的训练集和测试集
split_index = int(0.9 * len(features))

X_train = features_scaled[:split_index]
temp_y_train = temperature_labels[:split_index]
air_y_train = air_quality_labels[:split_index]
dates_train = features['date'][:split_index]


X_test = features_scaled[split_index:]
temp_y_test = temperature_labels[split_index:]
air_y_test = air_quality_labels[split_index:]
dates_test = features['date'][split_index:]


# 转换为 PyTorch 张量
X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
temp_y_train_tensor = torch.FloatTensor(temp_y_train.values)
temp_y_test_tensor = torch.FloatTensor(temp_y_test.values)
air_y_train_tensor = torch.FloatTensor(air_y_train.values)
air_y_test_tensor = torch.FloatTensor(air_y_test.values)

# 定义温度预测模型
class TemperaturePredictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(TemperaturePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 定义空气质量预测模型
class AirQualityPredictor(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(AirQualityPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 初始化温度预测模型
temp_input_size = len(features.columns)
temp_hidden_size1 = 128
temp_hidden_size2 = 64
temp_output_size = 2  # 输出为2个值，即最高温度和最低温度
temperature_model = TemperaturePredictor(temp_input_size, temp_hidden_size1, temp_hidden_size2, temp_output_size)

# 初始化空气质量预测模型
air_input_size = len(features.columns)
air_hidden_size1 = 128
air_hidden_size2 = 64
air_output_size = 1  # 输出为1个值，即空气质量
air_quality_model = AirQualityPredictor(air_input_size, air_hidden_size1, air_hidden_size2, air_output_size)

# 定义温度预测模型的损失函数和优化器
temp_criterion = nn.MSELoss()
temp_optimizer = optim.Adam(temperature_model.parameters(), lr=0.001)

# 定义空气质量预测模型的损失函数和优化器
air_criterion = nn.MSELoss()
air_optimizer = optim.Adam(air_quality_model.parameters(), lr=0.001)

# 训练温度预测模型
num_epochs = 600
for epoch in range(num_epochs):
    temp_outputs = temperature_model(X_train_tensor)
    temp_loss = temp_criterion(temp_outputs, temp_y_train_tensor)
    
    temp_optimizer.zero_grad()
    temp_loss.backward()
    temp_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Temperature Model - Epoch [{epoch+1}/{num_epochs}], Loss: {temp_loss.item():.4f}')

# 保存温度预测模型
torch.save(temperature_model.state_dict(), 'temperature_model.pth')

with torch.no_grad():
    temperature_model.eval()
    predictions = temperature_model(X_test_tensor)

# 将预测值和真实值转换为NumPy数组
predictions = predictions.numpy()
y_test_array = temp_y_test_tensor.numpy()


# 将训练集特征转换为NumPy数组
X_train_np = X_train_tensor.numpy()
y_train_np = temp_y_train_tensor.numpy()
# 在训练集上进行预测
with torch.no_grad():
    temperature_model.eval()
    train_predictions = temperature_model(X_train_tensor).numpy()




'''
# 训练空气质量预测模型
for epoch in range(5000):
    air_outputs = air_quality_model(X_train_tensor)
    air_loss = air_criterion(air_outputs, air_y_train_tensor)
    
    air_optimizer.zero_grad()
    air_loss.backward()
    air_optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Air Quality Model - Epoch [{epoch+1}/{num_epochs}], Loss: {air_loss.item():.4f}')
'''
# 在测试集上评估温度预测模型
with torch.no_grad():
    temperature_model.eval()
    temp_outputs = temperature_model(X_test_tensor)
    temp_mse = nn.MSELoss()(temp_outputs, temp_y_test_tensor)
    print(f'Test Temperature Mean Squared Error: {temp_mse.item():.4f}')
  
for i in range(10):
    Date = datetime.utcfromtimestamp(dates_test.iloc[i] * 24 * 3600)
    Date=Date.strftime('%Y-%m-%d')
    print(f'Sample {i + 1}: {Date}')
    print(f'   Predicted: High Temp: {predictions[i][0]:.2f}, Low Temp: {predictions[i][1]:.2f}')
    print(f'   Actual:    High Temp: {y_test_array[i][0]}, Low Temp: {y_test_array[i][1]}')
    print('\n')
'''
# 在测试集上评估空气质量预测模型
with torch.no_grad():
    air_quality_model.eval()
    air_outputs = air_quality_model(X_test_tensor)
    air_mse = nn.MSELoss()(air_outputs, air_y_test_tensor)
    print(f'Test Air Quality Mean Squared Error: {air_mse.item():.4f}')
'''


# 将日期从天数转换为日期对象
dates_test_datetime = [datetime.utcfromtimestamp(date * 24 * 3600) for date in dates_test]

# 绘制拟合曲线
plt.figure(figsize=(12, 6))

# 高温度拟合曲线
plt.subplot(1, 2, 1)
plt.scatter(dates_test_datetime, y_test_array[:, 0], label='Actual High Temp', color='blue')
plt.scatter(dates_test_datetime, temp_outputs[:, 0], label='Predicted High Temp', color='red')
plt.xlabel('Date')
plt.ylabel('High Temp')
plt.title('High Temp Fitting Curve - Test Set')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# 低温度拟合曲线
plt.subplot(1, 2, 2)
plt.scatter(dates_test_datetime, y_test_array[:, 1], label='Actual Low Temp', color='blue')
plt.scatter(dates_test_datetime, temp_outputs[:, 1], label='Predicted Low Temp', color='red')
plt.xlabel('Date')
plt.ylabel('Low Temp')
plt.title('Low Temp Fitting Curve - Test Set')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.tight_layout()
plt.show()

