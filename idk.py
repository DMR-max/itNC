import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Load the dataset
df = pd.read_csv('train.csv')
timeseries = df[["store", "product", "number_sold"]].values.astype('float32')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
timeseries = scaler.fit_transform(timeseries)

# Train-test split for time series
train_size = int(len(timeseries) * 0.67)
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback - 10):
        feature = dataset[i : i + lookback]
        target = dataset[i + lookback : i + lookback + 10]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)

lookback = 100
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

class RecurrentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 10)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = RecurrentNN()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=32)

n_epochs = 200
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    y_pred_train = model(X_train)
    y_pred_test = model(X_test)
    train_mape = mean_absolute_percentage_error(y_train.detach().numpy(), y_pred_train.detach().numpy())
    test_mape = mean_absolute_percentage_error(y_test.detach().numpy(), y_pred_test.detach().numpy())
    print("Train MAPE: %.4f, Test MAPE: %.4f" % (train_mape, test_mape))
