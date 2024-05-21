import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

st.set_page_config(
    page_title="测试",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
)
st.title("模型测试")

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

# 创建DataFrame保存测试结果
result_df = pd.DataFrame(columns=['Date', 'Predicted High Temp', 'Predicted Low Temp', 'Actual High Temp', 'Actual Low Temp'])

# 填充DataFrame
for i in range(len(predictions)):
    Date = datetime.utcfromtimestamp(features['date'].iloc[i + split_index] * 24 * 3600)
    Date = Date.strftime('%Y-%m-%d')
    predicted_high_temp = predictions[i][0]
    predicted_low_temp = predictions[i][1]
    actual_high_temp = y_test_array[i][0]
    actual_low_temp = y_test_array[i][1]

    result_df = result_df.append({'Date': Date, 'Predicted High Temp': predicted_high_temp,
                                  'Predicted Low Temp': predicted_low_temp, 'Actual High Temp': actual_high_temp,
                                  'Actual Low Temp': actual_low_temp}, ignore_index=True)

# 显示DataFrame
st.subheader("模型测试结果")
st.dataframe(result_df)

# 绘制拟合曲线
st.subheader("拟合曲线")
dates_test_datetime = [datetime.utcfromtimestamp(date * 24 * 3600) for date in features['date'][split_index:]]
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(dates_test_datetime, y_test_array[:, 0], label='Actual High Temp', color='blue')
ax.scatter(dates_test_datetime, predictions[:, 0], label='Predicted High Temp', color='red')
ax.set_xlabel('Date')
ax.set_ylabel('High Temp')
ax.set_title('High Temp Fitting Curve - Test Set')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
st.pyplot(fig)
