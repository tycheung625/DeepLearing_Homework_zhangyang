import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from streamlit.components.v1 import html
st.set_page_config(
    page_title="天气预测",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="auto",
)
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

# 初始化温度预测模型
temp_input_size = len(features.columns)
temp_hidden_size1 = 128
temp_hidden_size2 = 64
temp_output_size = 2  # 输出为2个值，即最高温度和最低温度
temperature_model = TemperaturePredictor(temp_input_size, temp_hidden_size1, temp_hidden_size2, temp_output_size)

# 定义温度预测模型的损失函数和优化器
temp_criterion = nn.MSELoss()
temp_optimizer = torch.optim.Adam(temperature_model.parameters(), lr=0.001)



# Streamlit App
st.title('模型训练')

# 交互式训练按钮
train_button = st.sidebar.button('训练温度预测模型')

# 创建表格的数据列表
table_data = []

# 创建一个空的 Matplotlib 图表容器
fig, ax = plt.subplots()
loss_curve, = ax.plot([], [], label='损失曲线')
ax.set_xlabel('Epoch')
ax.set_ylabel('损失')
ax.legend()

# 创建一个空的容器，用于显示 Matplotlib 图表
chart_container = st.container()

# 创建两列布局
col1, col2 = st.columns(2)
col1.info("Epochs loss:")
col2.info("Loss curve:")
# 创建表格的数据列表
table_data = []

if train_button:
	
    # 训练温度预测模型
    num_epochs = 600
    
    output_table = col1.dataframe()
    chart_container = col2.empty()
    for epoch in range(num_epochs):
        temp_outputs = temperature_model(X_train_tensor)
        temp_loss = temp_criterion(temp_outputs, temp_y_train_tensor)
        
        temp_optimizer.zero_grad()
        temp_loss.backward()
        temp_optimizer.step()

        if (epoch + 1) % 10 == 0:
            # 将训练结果合并成一个字符串，添加到表格数据列表
            row_data = f'Epoch [{epoch+1}/{num_epochs}]，损失：{temp_loss.item():.4f}'
            table_data.append(row_data)

            # 使用返回值更新表格数据源
            output_table.dataframe(table_data)

            # 更新损失曲线
            loss_curve.set_xdata(range(1, epoch + 2, 10))
            loss_curve.set_ydata([float(data.split('：')[-1]) for data in table_data])
            ax.relim()
            ax.autoscale_view()

            # 在第一列显示 Matplotlib 图表
            with chart_container:
                st.pyplot(fig)
    col1.success("训练完成")
