import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_util

import sklearn
from sklearn.model_selection import train_test_split # for preparing the training and testing data sets. You should get yourself familiar with it.
from sklearn.preprocessing import MinMaxScaler       # Data preprocessing
from sklearn.metrics import accuracy_score           # performance metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv('train.csv')
df.insert(0,'index_column','')
df['index_column'] = df.index
print(df)
timeseries = df[['index_column',"number_sold"]].values.astype('float32')


# Reduce the size of the dataset
# sample_size = 100
# if len(timeseries) > sample_size:
#     indices = np.random.choice(len(timeseries), size=sample_size, replace=False)
#     timeseries = timeseries[indices]

#plt.plot(timeseries)
# plt.show()

# Normalize the dataset
# timeseries = timeseries.reshape(-1,1) # column is 1 but row is unknown so numpy figures dimensions out
# timeseries = timeseries.astype("float32")
# timeseries.shape

# scaler = MinMaxScaler(feature_range=(0, 1))
# timeseries = scaler.fit_transform(timeseries)

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
        self.lstm = nn.LSTM(input_size=2, hidden_size=4, num_layers=1, batch_first=True)
        self.linear = nn.Linear(4,2)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

# train and validate the RNN model
model = RecurrentNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.MSELoss()

# new concept: the Data Loader, which is handy for converting the data (in numpy or pandas dataframes) to Tensors
# please check out this very detailed tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
loader = data_util.DataLoader(data_util.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch) #moet naat MAPE
        #loss = mean_absolute_percentage_error(y_batch, y_pred)
        optimizer.zero_grad() # cleanzing the gradient information from the previous step/iteration
        loss.backward()       # compute the gradients by backpropagation
        optimizer.step()      # gradient descent

    # validate the model on the test set - only performed after some epochs
    if epoch % 100 != 0:
        continue
    # this line is essential for validation: it turns off some functionalities that should not active in testing, e.g., dropout
    model.eval()
    # this line is essentail for validation: it temporarily disables the computation of the gradient in the backpropagation.
    with torch.no_grad():
        y_pred = model(X_train)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train))
        y_pred = model(X_test)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test))

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))



df = pd.read_csv("test_example.csv")
window_size = 50  # please fill in your own choice: this is the length of history you have to decide

# split the data set by the combination of `store` and `product``
gb = df.groupby(["store", "product"])
groups = {x: gb.get_group(x) for x in gb.groups}
scores = {}

for key, data in groups.items():
    # By default, we only take the column `number_sold`.
    # Please modify this line if your model takes other columns as input
    X = data.drop(["Date","store", "product"], axis=1).values  # convert to numpy array
    N = X.shape[0]  # total number of testing time steps

    mape_score = []
    start = window_size
    # prediction by window rolling
    while start + 50 <= N:
        inputs = X[(start - window_size) : start, :]
        targets = X[start : (start + 50), :]
        print(inputs)

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