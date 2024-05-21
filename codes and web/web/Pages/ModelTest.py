import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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

@st.cache_resource
def plotlinear():
	
	data = pd.read_csv('beijing2.csv')

	# 提取特征和标签
	features = data[['date', '最高温度', '最低温度']]
	temperature_labels = data[['最高温度', '最低温度']]

	
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
		temp_mse = nn.MSELoss()(predictions, temp_y_test_tensor)
		st.write(f'Test Temperature Mean Squared Error: {temp_mse.item():.4f}')

	
	predictions = predictions.numpy()
	y_test_array = temp_y_test_tensor.numpy()

	
	result_df = pd.DataFrame(columns=['Date', 'Predicted High Temp', 'Predicted Low Temp', 'Actual High Temp', 'Actual Low Temp'])

	
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

@st.cache_resource
def plot_rnn():
	
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
			#date = data[date_index, 0]  # 假设日期是数组中的第一列，根据实际情况调整
			date = data[date_index,0]
			sequences.append(seq)
			targets.append(target)
			dates.append(date)
		return torch.tensor(sequences), torch.tensor(targets), dates
    
	seq_length = 15

	X_train, y_train, train_dates = create_sequences(train_data, seq_length)
	X_test, y_test, test_dates = create_sequences(test_data, seq_length)



	
	input_size = len(target_cols)
	hidden_size = 128
	output_size = len(target_cols)
	num_layers = 3

	
	model = RNN(input_size, hidden_size, output_size, num_layers)

	
	model.load_state_dict(torch.load('rnn_model.pth')) 
	model.eval()
	# 测试模型
	model.eval()
	with torch.no_grad():
		test_outputs = model(X_test.float())
		mse = nn.MSELoss()(test_outputs, y_test.float())
		st.write(f'Mean Squared Error on Test Data: {mse.item():.4f}')



	
	test_dates_original = data.index[train_size + seq_length:]
	predictions_df = pd.DataFrame(test_outputs, columns=['Predicted High Temp', 'Predicted Low Temp'], index=test_dates_original)
	actuals_df = pd.DataFrame(y_test, columns=['Actual High Temp', 'Actual Low Temp'], index=test_dates_original)

	result_df = pd.concat([predictions_df, actuals_df], axis=1)

	result_df.index.name = 'Date'
	# 格式化DataFrame的日期为"年-月-日"形式
	result_df.index = result_df.index.strftime('%Y-%m-%d')

	
	st.subheader("模型测试结果")
	st.dataframe(result_df)

	# 绘制最高温度和最低温度拟合曲线
	st.subheader("拟合曲线")
	fig, ax = plt.subplots(figsize=(12, 6))

	# 绘制最高温度拟合曲线
	ax.plot(test_dates_original, y_test[:, 0], label='Actual High Temperature', marker='o', color='blue')
	ax.plot(test_dates_original, test_outputs[:, 0], label='Predicted High Temperature', marker='o', linestyle='dashed', color='orange')

	# 绘制最低温度拟合曲线
	ax.plot(test_dates_original, y_test[:, 1], label='Actual Low Temperature', marker='o', color='green')
	ax.plot(test_dates_original, test_outputs[:, 1], label='Predicted Low Temperature', marker='o', linestyle='dashed', color='red')

	ax.set_xlabel('Date')
	ax.set_ylabel('Temperature')
	ax.set_title('Temperature Prediction')
	ax.legend()

	# 在Streamlit中显示图表
	st.pyplot(fig)

class upname:
    def __init__(self, name):
        self.name = name
if __name__ == '__main__':
	option=st.sidebar.selectbox('方法:',('Linear','RNN'))
	if option=='Linear':
		st.info("模型名称：Linear")
		plotlinear()
	if option=='RNN':
		st.info("模型名称：RNN")
		plot_rnn()
