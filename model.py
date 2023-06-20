# we first import some dependencies
import sklearn
from sklearn.model_selection import train_test_split # for preparing the training and testing data sets. You should get yourself familiar with it.
from sklearn.preprocessing import MinMaxScaler       # Data preprocessing
from sklearn.metrics import accuracy_score           # performance metrics
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

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

data_num_sold = df['number_sold'].values
data_product = df['product'].values
plt.plot(data_num_sold, data_product,color= 'blue')
plt.xlabel("products")
plt.ylabel("Number sold")
plt.title("International airline passengers")
plt.show()

# Normalize the dataset
data = data_num_sold.reshape(-1,1) # column is 1 but row is unknown so numpy figures dimensions out
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
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)
for i in range(len(train)-time_step):
    feature = train[i:i+time_step]
    target = train[i+1:i+time_step+1]
    xtrain = np.append(xtrain, feature)
    ytrain = np.append(ytrain, target)
    trainX = torch.tensor(xtrain)
    trainY = torch.tensor(ytrain)

xtest, ytest = [], []
xtest = np.array(xtest)
ytest = np.array(ytest)
for i in range(len(test)-time_step):
    feature = test[i:i+time_step]
    target = test[i+1:i+time_step+1]
    xtest = np.append(xtest, feature)
    ytest = np.append(ytest, target)
    testX = torch.tensor(xtest)
    testY = torch.tensor(ytest)

print(trainX.shape, trainY.shape)
print(testX.shape, testY.shape)

class RecurrentNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_unit = torch.nn.Module.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        # input = torch.randn(5, 3, 10)
        # h0 = torch.randn(2, 3, 20)
        # c0 = torch.randn(2, 3, 20)
        
        # please create an LSTM unit with the build-in module `torch.nn.LSTM`.
                        # You can decide on your own the dimension/size of the hidden state and the number of layers for LSTM
                        # please check the official documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
        
        self.output_unit = torch.nn.Linear(50, 1)
        # self.rnn_unit(input, (h0, c0)) # which unit we should use here? Remember we are supposed to forcast the next (t+1) data point from the hidden state/cell state of LSTM
        

    def forward(self, x: torch.Tensor):
        output,_ = self.rnn_unit(x)
        output = self.output_unit(output)
        return output
    
# train and validate the RNN model
model = RecurrentNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.MSELoss()

df = pd.read_csv("test_example.csv")
model = None  # load your model here
window_size = 10  # please fill in your own choice: this is the length of history you have to decide

# split the data set by the combination of `store` and `product``
gb = df.groupby(["store", "product"])
groups = {x: gb.get_group(x) for x in gb.groups}
scores = {}

for key, data in groups.items():
    # By default, we only take the column `number_sold`.
    # Please modify this line if your model takes other columns as input
    X = data.drop(["Date", "store", "product"], axis=1).values  # convert to numpy array
    N = X.shape[0]  # total number of testing time steps

    mape_score = []
    start = window_size
    # prediction by window rolling
    while start + 5 <= N:
        inputs = X[(start - window_size) : start, :]
        targets = X[start : (start + 5), :]

        # you might need to modify `inputs` before feeding it to your model, e.g., convert it to PyTorch Tensors
        # you might have a different name of the prediction function. Please modify accordingly
        predictions = model.predict(inputs)
        start += 5
        # calculate the performance metric
        mape_score.append(mean_absolute_percentage_error(targets, predictions))
    scores[key] = mape_score

# save the performance metrics to file
np.savez("score.npz", scores=scores)

