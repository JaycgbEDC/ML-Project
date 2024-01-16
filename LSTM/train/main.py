import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import torch

filepath1 = 'D:\QMDownload\Microsoft Edge Loader\机器学习-作业\ML_RL\LSTM\ETDataset-main\ETT-small\Train_set.csv'
filepath2 = 'D:\QMDownload\Microsoft Edge Loader\机器学习-作业\ML_RL\LSTM\ETDataset-main\ETT-small\Test_set.csv'
data1 = pd.read_csv(filepath1)
data2 = pd.read_csv(filepath2)
data1 = data1.sort_values('date')
data2 = data2.sort_values('date')
print(data1.head())
print(data1.shape)

# def set_seed(seed):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

sns.set_style("darkgrid")
plt.figure(figsize=(15, 9))
plt.plot(data1[['OT']])
plt.xticks(range(0, data1.shape[0], 20), data1['date'].loc[::20], rotation=45)
plt.title("Stock Price", fontsize=18, fontweight='bold')
plt.xlabel('Date', fontsize=18)
plt.ylabel('OT', fontsize=18)
plt.show()

# 1.特征工程
# 选取Close作为特征
price1 = data1[['OT']]
price2 = data2[['OT']]
print(price1.info())
print(price2.info())

from sklearn.preprocessing import MinMaxScaler
# 进行不同的数据缩放，将数据缩放到-1和1之间
scaler = MinMaxScaler(feature_range=(-1, 1))
price1['OT'] = scaler.fit_transform(price1['OT'].values.reshape(-1, 1))
print(price1['OT'].shape)
price2['OT'] = scaler.fit_transform(price2['OT'].values.reshape(-1, 1))
print(price2['OT'].shape)

# set_seed(5)

# 2.数据集制作
# 今天的收盘价预测明天的收盘价
# lookback表示观察的跨度
def split_data(stock1, stock2, lookback):
    data_raw1 = stock1.to_numpy()
    data_raw2 = stock2.to_numpy()
    train_data = []
    test_data = []
    # print(data)

    # you can free play（seq_length）
    for index in range(len(data_raw1) - lookback):
        train_data.append(data_raw1[index: index + lookback])
    for index in range(len(data_raw2) - lookback):
        test_data.append(data_raw2[index: index + lookback])

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    x_train = train_data[0:, :-1]
    y_train = train_data[0:, -1, :]

    x_test = test_data[0:, :-1]
    y_test = test_data[0:, -1, :]

    return [x_train, y_train, x_test, y_test]

lookback = 96
# lookback = 336

x_train, y_train, x_test, y_test = split_data(price1, price2, lookback)

print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

# 注意：pytorch的nn.LSTM input shape=(seq_length, batch_size, input_size)
# 3.模型构建 —— LSTM

import torch
import torch.nn as nn

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)
y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
y_train_gru = torch.from_numpy(y_train).type(torch.Tensor)
y_test_gru = torch.from_numpy(y_test).type(torch.Tensor)
# 输入的维度为1，只有Close收盘价
input_dim = 1
# 隐藏层特征的维度
hidden_dim = 32
# 循环的layers
num_layers = 2
# 预测后一天的收盘价
output_dim = 1
num_epochs = 100


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out



model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
# optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
# optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
# optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)

# 4.模型训练
import time

hist = np.zeros(num_epochs)
start_time = time.time()
lstm = []

for t in range(num_epochs):
    y_train_pred = model(x_train)

    loss = criterion(y_train_pred, y_train_lstm)
    # print("Epoch ", t, "MSE: ", loss.item())
    print("Epoch ", t, "MAE: ", loss.item())
    hist[t] = loss.item()

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

training_time = time.time() - start_time
print("Training time: {}".format(training_time))

# 5.模型结果可视化

predict = pd.DataFrame(scaler.inverse_transform(y_train_pred.detach().numpy()))
original = pd.DataFrame(scaler.inverse_transform(y_train_lstm.detach().numpy()))

import seaborn as sns
sns.set_style("darkgrid")

fig = plt.figure()
fig.subplots_adjust(hspace=0.2, wspace=0.2)

plt.subplot(1, 2, 1)
ax = sns.lineplot(x = original.index, y = original[0], label="Data", color='royalblue')
ax = sns.lineplot(x = predict.index, y = predict[0], label="Training Prediction (LSTM)", color='tomato')
# print(predict.index)
# print(predict[0])


ax.set_title('Stock price', size = 14, fontweight='bold')
ax.set_xlabel("Days", size = 14)
ax.set_ylabel("Cost", size = 14)
ax.set_xticklabels('', size=10)


plt.subplot(1, 2, 2)
ax = sns.lineplot(data=hist, color='royalblue')
ax.set_xlabel("Epoch", size = 14)
ax.set_ylabel("Loss", size = 14)
ax.set_title("Training Loss", size = 14, fontweight='bold')
fig.set_figheight(6)
fig.set_figwidth(16)
plt.show()


# 6.模型验证
# print(x_test[-1])
import math, time
from sklearn.metrics import mean_squared_error

# make predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train_lstm.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test_lstm.detach().numpy())

# calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMAE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMAE' % (testScore))
lstm.append(trainScore)
lstm.append(testScore)
lstm.append(training_time)

# In[40]:

price = pd.concat([price1, price2], axis=0)
print(np.std(price2))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(price)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[lookback:len(y_train_pred)+lookback, :] = y_train_pred

# shift test predictions for plotting
testPredictPlot = np.empty_like(price)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(y_train_pred)+lookback-1:len(price)-lookback-1, :] = y_test_pred

original = scaler.inverse_transform(price['OT'].values.reshape(-1,1))

predictions = np.append(trainPredictPlot, testPredictPlot, axis=1)
predictions = np.append(predictions, original, axis=1)
result = pd.DataFrame(predictions)

import plotly.express as px
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[0],
                    mode='lines',
                    name='Train prediction')))
fig.add_trace(go.Scatter(x=result.index, y=result[1],
                    mode='lines',
                    name='Test prediction'))
fig.add_trace(go.Scatter(go.Scatter(x=result.index, y=result[2],
                    mode='lines',
                    name='Actual Value')))
fig.update_layout(
    xaxis=dict(
        showline=True,
        showgrid=True,
        showticklabels=False,
        linecolor='black',
        linewidth=2
    ),
    yaxis=dict(
        title_text='OT',
        titlefont=dict(
            family='Rockwell',
            size=12,
            color='black',
        ),
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='black',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Rockwell',
            size=12,
            color='black',
        ),
    ),
    showlegend=True,
    template = 'plotly_white'

)



annotations = []
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Results (LSTM)',
                              font=dict(family='Rockwell',
                                        size=26,
                                        color='white'),
                              showarrow=False))
fig.update_layout(annotations=annotations)

fig.show()