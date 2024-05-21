import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y, depth=0):
        if depth == self.max_depth or len(np.unique(y, axis=0)) == 1 or len(X) < self.min_samples_split:
            self.tree = {'value': np.mean(y, axis=0)}
            return self.tree

        num_features = X.shape[1]
        feature_index = np.random.choice(num_features, 1)[0]
        feature_values = X[:, feature_index]

        median_value = np.median(feature_values)

        left_mask = feature_values <= median_value
        right_mask = ~left_mask

        left_subtree = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self.fit(X[right_mask], y[right_mask], depth + 1)

        self.tree = {'left': left_subtree, 'right': right_subtree, 'index': feature_index, 'value': median_value}
        return self.tree

    def predict(self, X):
        if 'value' in self.tree:
            return np.full((X.shape[0], len(self.tree['value'])), self.tree['value'])

        feature_value = X[:, self.tree['index']]

        left_mask = feature_value <= self.tree['value']
        right_mask = ~left_mask

        left_predictions = self.predict(X[left_mask])
        right_predictions = self.predict(X[right_mask])

        predictions = np.empty((X.shape[0], len(self.tree['value'])))
        predictions[left_mask] = left_predictions
        predictions[right_mask] = right_predictions

        return predictions
'''
class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        num_features = X.shape[1]

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            indices = np.random.choice(num_features, self.max_features, replace=False)
            tree.fit(X[:, indices], y)
            self.trees.append({'tree': tree, 'indices': indices})

    def predict(self, X):
        def tree_predict(tree, X):
            return tree['tree'].predict(X[:, tree['indices']]) if 'value' not in tree['tree'] else np.full((X.shape[0], len(tree['tree']['value'])), tree['tree']['value'])

        predictions = np.apply_along_axis(lambda tree: tree_predict(tree, X), axis=0, arr=self.trees)
        return np.mean(predictions, axis=0)
'''
class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        num_features = X.shape[1]

        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            indices = np.random.choice(num_features, self.max_features, replace=False)
            tree.fit(X[:, indices], y)
            self.trees.append({'tree': tree, 'indices': indices})

    def predict(self, X):
        predictions = np.zeros((X.shape[0], 2))  # Assuming 2 output values (高温 and 低温)

        for _, tree_info in enumerate(self.trees):
            tree = tree_info['tree']
            indices = tree_info['indices']

            tree_predictions = tree.predict(X[:, indices])
            if tree_predictions.ndim == 1:
                tree_predictions = np.expand_dims(tree_predictions, axis=1)

            predictions += tree_predictions

        return predictions / self.n_trees






# 读取数据
data = pd.read_csv('beijing2.csv')

# 提取特征和标签
features = data[['date', '最高温度', '最低温度']]
labels = data[['最高温度', '最低温度']]

# 将日期转换为天数，作为一个简单的特征
features['date'] = features['date'].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - datetime(2019, 1, 1)).days)

# 数据预处理
features_scaled = features.values
labels_scaled = labels.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_scaled, test_size=0.2, random_state=42)

# 初始化和训练随机森林模型
rf_model = RandomForest(n_trees=100, max_depth=15, min_samples_split=5, max_features=3)
rf_model.fit(X_train, y_train)

# 在测试集上进行预测
rf_predictions = rf_model.predict(X_test)

# 计算均方误差
rf_mse = mean_squared_error(y_test, rf_predictions)
print(f'Random Forest Mean Squared Error: {rf_mse:.4f}')
