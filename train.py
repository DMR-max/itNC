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

class RecurrentNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_unit = torch.nn.Module.LSTM(10, 20, 2) 
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)
        
        # please create an LSTM unit with the build-in module `torch.nn.LSTM`.
                        # You can decide on your own the dimension/size of the hidden state and the number of layers for LSTM
                        # please check the official documentation: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html 
        
        self.output_unit = self.rnn_unit(input, (h0, c0)) # which unit we should use here? Remember we are supposed to forcast the next (t+1) data point from the hidden state/cell state of LSTM
        

    def forward(self, x: torch.Tensor):
        output,_ = self.rnn_unit(x)
        output = self.output_unit(output)
        return output
    
# train and validate the RNN model
model = RecurrentNN()

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.MSELoss()

# new concept: the Data Loader, which is handy for converting the data (in numpy or pandas dataframes) to Tensors
# please check out this very detailed tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
loader = data_util.DataLoader(data_util.TensorDataset(trainX, trainY), shuffle=True, batch_size=8)
 
n_epochs = 2000
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
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
        y_pred = model(trainX)
        train_rmse = np.sqrt(loss_fn(y_pred, trainY))
        y_pred = model(testX)
        test_rmse = np.sqrt(loss_fn(y_pred, testY))

    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

