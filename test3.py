import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from datetime import datetime

# 读取数据
data = pd.read_csv('beijing2.csv')

# 提取特征和标签
features = data[['date', '最高温度']]
labels = data['最低温度']

# 将日期转换为天数，作为一个简单的特征
features['date'] = features['date'].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime(2019, 1, 1)).days)

# 数据预处理
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 添加多项式特征
poly_degree = 11
poly_features = PolynomialFeatures(degree=poly_degree, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 设置训练轮数列表
num_epochs_list = [100, 500, 1000,10000]

# 设置 alpha 值列表
alpha_values = [0.0000001,0.001, 0.01, 0.1, 1, 10]

# 训练模型多次并输出每次的性能
for num_epochs in num_epochs_list:
    for alpha in alpha_values:
        # 初始化 Ridge Regression 模型
        ridge_model = Ridge(alpha=alpha, max_iter=num_epochs)

        # 训练模型
        ridge_model.fit(X_train_poly, y_train)

        # 在测试集上进行预测
        ridge_predictions = ridge_model.predict(X_test_poly)

        # 计算均方误差
        ridge_mse = mean_squared_error(y_test, ridge_predictions)

        # 输出性能结果
        print(f'Ridge Regression (Alpha={alpha}, Epochs={num_epochs}) Mean Squared Error: {ridge_mse:.4f}')
