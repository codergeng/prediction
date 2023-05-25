import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

# 假设我们有一个名为'data'的DataFrame，且我们对最后一列进行预测
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :-1]
        dataX.append(a)
        dataY.append(dataset[i + look_back, -1])
    return np.array(dataX), np.array(dataY)

# 滑动窗口大小
look_back = 3

# 为LSTM准备数据
data = pd.DataFrame(...)  # 这里使用你的数据
data = data.values
data = data.astype('float32')

# 划分数据为训练集和测试集
train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]

# 创建滑动窗口
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# 调整输入数据的维度以适应LSTM (samples, timesteps, features)
trainX = np.reshape(trainX, (trainX.shape[0], look_back, trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], look_back, testX.shape[2]))

# 创建并训练LSTM网络
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# 预测
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 计算和打印评估指标
trainScore = mean_absolute_error(trainY, trainPredict)
print('Train Score: %.2f MAE' % (trainScore))
trainScore = sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = mean_absolute_error(testY, testPredict)
print('Test Score: %.2f MAE' % (testScore))
testScore = sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))
