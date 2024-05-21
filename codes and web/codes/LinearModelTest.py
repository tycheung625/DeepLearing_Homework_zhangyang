import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# 读取数据
data = pd.read_csv('beijing2.csv')

# 提取特征和标签
features = data[['date', '最高温度', '最低温度']]
temperature_labels = data[['最高温度', '最低温度']]

# 将日期转换为天数，作为一个简单的特征
features['date'] = features['date'].apply(lambda x: (datetime.strptime(x, "%Y/%m/%d") - datetime(1970, 1, 1)).days)

# 数据预处理
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 选择固定的训练集和测试集
split_index = int(0.9 * len(features))

# 转换为 PyTorch 张量
X_test_tensor = torch.FloatTensor(features_scaled[split_index:])
temp_y_test_tensor = torch.FloatTensor(temperature_labels.values[split_index:])

# 加载温度预测模型
loaded_temperature_model = TemperaturePredictor(input_size=3, hidden_size1=128, hidden_size2=64, output_size=2)
loaded_temperature_model.load_state_dict(torch.load('temperature_model.pth'))
loaded_temperature_model.eval()

# 在测试集上进行预测
with torch.no_grad():
    predictions = loaded_temperature_model(X_test_tensor)

# 将预测值和真实值转换为NumPy数组
predictions = predictions.numpy()
y_test_array = temp_y_test_tensor.numpy()

# 打印测试结果
for i in range(10):
    Date = datetime.utcfromtimestamp(features['date'].iloc[i + split_index] * 24 * 3600)
    Date = Date.strftime('%Y-%m-%d')
    print(f'Sample {i + 1}: {Date}')
    print(f'   Predicted: High Temp: {predictions[i][0]:.2f}, Low Temp: {predictions[i][1]:.2f}')
    print(f'   Actual:    High Temp: {y_test_array[i][0]}, Low Temp: {y_test_array[i][1]}')
    print('\n')

# 绘制拟合曲线
dates_test_datetime = [datetime.utcfromtimestamp(date * 24 * 3600) for date in features['date'][split_index:]]
plt.figure(figsize=(12, 6))
plt.scatter(dates_test_datetime, y_test_array[:, 0], label='Actual High Temp', color='blue')
plt.scatter(dates_test_datetime, predictions[:, 0], label='Predicted High Temp', color='red')
plt.xlabel('Date')
plt.ylabel('High Temp')
plt.title('High Temp Fitting Curve - Test Set')
plt.legend()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.show()
