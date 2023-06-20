import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

df = pd.read_csv('train.csv')
timeseries = df[["store", "product","number_sold"]].values.astype('float32')

# Reduce the size of the dataset
sample_size = 100
if len(timeseries) > sample_size:
    indices = np.random.choice(len(timeseries), size=sample_size, replace=False)
    timeseries = timeseries[indices]
 
#plt.plot(timeseries)
# plt.show()

# train-test split for time series
train_size = int(len(timeseries) * 0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]



def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i : i + lookback]
        target = dataset[i + 1 : i + lookback + 1]
        X.append(feature)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X), torch.tensor(y)



lookback = 10
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)


 
class RecurrentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x



model = RecurrentNN()
df = pd.read_csv("test_example.csv")
window_size = 5  # please fill in your own choice: this is the length of history you have to decide

# split the data set by the combination of `store` and `product``
gb = df.groupby(["store", "product"])
groups = {x: gb.get_group(x) for x in gb.groups}
scores = {}

for key, data in groups.items():
    # By default, we only take the column `number_sold`.
    # Please modify this line if your model takes other columns as input
    X = data.drop(["Date"], axis=1).values  # convert to numpy array
    N = X.shape[0]  # total number of testing time steps

    mape_score = []
    start = window_size
    # prediction by window rolling
    while start + 5 <= N:
        inputs = X[(start - window_size) : start, :]
        targets = X[start : (start + 5), :]

        # you might need to modify `inputs` before feeding it to your model, e.g., convert it to PyTorch Tensors
        # you might have a different name of the prediction function. Please modify accordingly
        inputs = torch.tensor(inputs, dtype = torch.float32)
        predictions = model.forward(inputs)
        start += 5
        predictions = predictions.detach().numpy()

        # calculate the performance metric
        mape_score.append(mean_absolute_percentage_error(targets, predictions))
    scores[key] = mape_score
    print(scores)

# save the performance metrics to file
np.savez("score.npz", scores=scores)