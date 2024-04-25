import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error



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



# Perform testing and calculate performance metrics for the given model
def test_func(scaler, lookback, model):

    # Read the dataset from CSV file
    df = pd.read_csv("test_example.csv")
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y%m%d")
    window_size = 5

    # Split the dataset by the combination of 'store' and 'product'
    gb = df.groupby(["store", "product"])
    groups = {x: gb.get_group(x) for x in gb.groups}
    scores = {}

    for key, _ in groups.items():

        X = df[["Date", "store", "product", "number_sold"]].values.astype('float32') # convert to numpy array
        X = X.reshape(-1, 1) # column is 1 but row is unknown so numpy figures dimensions out
        X = X.astype("float32")
        X.shape
        Z = df[["number_sold"]].values.astype('float32')
        X = scaler.transform(X)
        Z = scaler.transform(Z)
        N = X.shape[0]  # total number of testing time steps
        X_train, _ = create_dataset(X, lookback=lookback)
        mape_score = []
        start = window_size

        # Prediction by window rolling
        while start + 5 <= N:
            X_inputs = X_train[(start - window_size):start, :]
            targets = Z[start:(start + 5), :]

            if len(targets) == 0 or len(X_inputs) == 0:
                break

            predictions = model.forward(X_inputs)
            start += 5
            predictions = torch.mean(predictions, dim=1)
            predictions = predictions.detach().numpy()

            # Calculate the performance metric
            mape_score.append(mean_absolute_percentage_error(targets, predictions))

        scores[key] = mape_score

    # Save the performance metrics to file
    np.savez("score.npz", scores=scores)
