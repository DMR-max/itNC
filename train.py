import test
import model

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_util
import copy
from sklearn.preprocessing import MinMaxScaler



# Calculate Mean Absolute Percentage Error (MAPE)
def loss_fn(output, target, epsilon=1e-7):

    return torch.mean(torch.abs((target - output) / target + epsilon))



# Transform a time series into a prediction dataset
def create_dataset(dataset, lookback):

    X, y = [], []
    for i in range(len(dataset) - lookback - 2):

        if (i % 4 == 0):
            feature1 = dataset[i : i + lookback]
        elif (i % 4 == 1):
            feature2 = dataset[i + 1 : i + lookback + 1]
        elif (i % 4 == 2):
            feature3 = dataset[i + 2 : i + lookback + 2]
            feature = np.column_stack((feature1, feature2, feature3))
            X.append(feature)
        elif (i % 4 == 3):
            target = dataset[i + 3 : i + lookback + 3]
            y.append(target)

    X = np.array(X)
    y = np.array(y)

    return torch.tensor(X), torch.tensor(y)



# Set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

# Read the dataset from CSV file
df = pd.read_csv('train.csv')
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d")
timeseries = df[["Date", "store", "product", "number_sold"]].values.astype('float32')

# Normalize the dataset
timeseries = timeseries.reshape(-1, 1)
timeseries = timeseries.astype("float32")

# Train-test split for time series
train_size = int(len(timeseries) * 0.70)
test_size = len(timeseries) - train_size
train_arr, test_arr = timeseries[:train_size], timeseries[train_size:]

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 2))
scaler.fit(train_arr)
train_arr = scaler.transform(train_arr)
test_arr = scaler.transform(test_arr)

lookback = 10
X_train, y_train = create_dataset(train_arr, lookback=lookback)
X_test, y_test = create_dataset(test_arr, lookback=lookback)

# Create the RNN model
model = model.RecurrentNN()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Create the data loader
loader = data_util.DataLoader(data_util.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

best_mape = np.inf
best_weights = None

n_epochs = 10
for epoch in range(n_epochs):
    model.train()  # Set the model in training mode

    for X_batch, y_batch in loader:
        y_pred = model(X_batch)  # Forward pass
        loss = loss_fn(y_pred, y_batch)  # Compute the loss
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

    model.eval()  # Set the model in evaluation mode

    with torch.no_grad():
        y_pred = model(X_test)  # Forward pass on the test set
        current_mape = float(loss_fn(y_pred, y_test))  # Compute the loss on the test set

        if current_mape < best_mape:
            best_mape = current_mape  # Update the best MAPE if the current MAPE is better
            best_weights = copy.deepcopy(model.state_dict())  # Store the best model weights

    print("Epoch %d: current MAPE %.6f, best MAPE %.6f" % (epoch + 1, current_mape, best_mape))

# Restore the best model and save its state
model.load_state_dict(best_weights)
torch.save(model.state_dict(), "model_state_dict.pth")

# Print the best MAPE achieved
print("Best MAPE: %.6f" % best_mape)

# Call the test function
test.test_func(scaler, lookback, model)
