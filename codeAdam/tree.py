from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class RandomForestRegressor:
    """
    随机森林回归器
    """
    
    def __init__(self, n_estimators=10, random_state=0):
        # 随机森林的大小
        self.n_estimators = n_estimators
        # 随机森林的随机种子
        self.random_state = random_state
        # 决策树数组
        self.trees = []

    def fit(self, X, y):
        """
        随机森林回归器拟合
        """
        n = X.shape[0]
        rs = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            # 创建决策树回归器
            dt = DecisionTreeRegressor(random_state=rs.randint(np.iinfo(np.int32).max), max_features=1.0)
            # 根据随机生成的权重，拟合数据集
            indices = rs.randint(0, n, n)
            dt.fit(X.iloc[indices], y.iloc[indices])
            self.trees.append(dt)

    def predict(self, X):
        """
        随机森林回归器预测
        """
        # 预测结果
        ys = np.zeros((X.shape[0], self.n_estimators, 2))  
        for i in range(self.n_estimators):
            # 决策树回归器
            dt = self.trees[i]
            # 依次预测结果
            ys[:, i, :] = dt.predict(X)
        # 预测结果取平均
        predictions = np.mean(ys, axis=1)
        return predictions




# 读取CSV文件
file_path = 'beijing2.csv'
data = pd.read_csv(file_path)

# 将日期列转换为天数
data['days_since_reference'] = (pd.to_datetime(data['date'], format="%Y/%m/%d") - datetime(1970, 1, 1)).dt.days

# 选择特征和目标变量
features = data[['days_since_reference']]
targets = data[['最高温度', '最低温度']]

# 创建随机森林回归器实例
rf_regressor = RandomForestRegressor(n_estimators=10, random_state=0)

# 使用 fit 方法拟合模型
rf_regressor.fit(features, targets)

# 读取待预测的CSV文件
file_path_new_data = 'test.csv'
prediction_data = pd.read_csv(file_path_new_data)
# 将日期列转换为天数
prediction_data['days_since_reference'] = (pd.to_datetime(prediction_data['date'], format="%Y/%m/%d") - datetime(1970, 1, 1)).dt.days

features_new_data = prediction_data[['days_since_reference']]  

# 使用拟合完成的模型进行预测
predictions = rf_regressor.predict(features_new_data)

# 打印预测值
#print("预测值：")
#print(predictions)


plt.figure(figsize=(10, 6))

# 转换日期格式
date_format = mdates.DateFormatter('%Y/%m/%d')

# 找到真实值和预测值的共同横坐标范围
min_date = max(data['days_since_reference'].min(), prediction_data['days_since_reference'].min())
max_date = min(data['days_since_reference'].max(), prediction_data['days_since_reference'].max())

# 筛选真实值在共同横坐标范围内的数据
filtered_real_data = data[(data['days_since_reference'] >= min_date) & (data['days_since_reference'] <= max_date)]

# 绘制真实值
plt.plot(mdates.num2date(filtered_real_data['days_since_reference']), filtered_real_data['最高温度'], label='MaxTemp_Real', color='blue')
plt.plot(mdates.num2date(filtered_real_data['days_since_reference']), filtered_real_data['最低温度'], label='MinTemp_Real', color='green')

# 绘制预测值
plt.plot(mdates.num2date(prediction_data['days_since_reference']), predictions[:, 0], label='MaxTemp_Predict', linestyle='dashed', color='red')
plt.plot(mdates.num2date(prediction_data['days_since_reference']), predictions[:, 1], label='MinTemp_Predict', linestyle='dashed', color='orange')

# 设置横坐标的日期格式
plt.gca().xaxis.set_major_formatter(date_format)

plt.xlabel('DayTime')
plt.ylabel('Temp')
plt.title('Result')
plt.legend()
plt.show()