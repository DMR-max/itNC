import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

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