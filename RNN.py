# we first import some dependencies
import sklearn
from sklearn.model_selection import train_test_split # for preparing the training and testing data sets. You should get yourself familiar with it.
from sklearn.preprocessing import MinMaxScaler       # Data preprocessing
from sklearn.metrics import accuracy_score           # performance metrics
import matplotlib.pyplot as plt

import torch                                         # ofc, the PyTorch library
import torch.optim as optim
import torch.utils.data as data_util

import numpy as np
import pandas as pd

# set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

df = pd.read_csv("train.csv")
df.head()

data = df['number_sold'].values
plt.plot(data,color= 'blue')
plt.xlabel("Time")
plt.ylabel("Number sold")
plt.title("International airline passengers")
plt.show()

# Normalize the dataset
data = data.reshape(-1,1)
data = data.astype("float32")
data.shape

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Split the time series dataset into train and test sets
train_size = int(len(data) * 0.70)
test_size = len(data) - train_size
train, test = data[:train_size], data[train_size:]
print("train size: {}, test size: {} ".format(len(train), len(test)))

# Data transformation to tensors

time_step = 20

xtrain, ytrain = [], []
for i in range(len(train)-time_step):
    feature = train[i:i+time_step]
    target = train[i+1:i+time_step+1]
    xtrain.append(feature)
    ytrain.append(target)
    trainX = torch.tensor(xtrain)
    trainY = torch.tensor(ytrain)

xtest, ytest = [], []
for i in range(len(test)-time_step):
    feature = test[i:i+time_step]
    target = test[i+1:i+time_step+1]
    xtest.append(feature)
    ytest.append(target)
    testX = torch.tensor(xtest)
    testY = torch.tensor(ytest)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)