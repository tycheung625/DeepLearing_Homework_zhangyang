import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 加载数据
data = pd.read_csv('beijing2.csv')
data['date'] = pd.to_datetime(data['date'])
#print(data['date',0])
data = data.set_index('date')
print(data[0])
