We do some preprocessing of the dataset, first we reshape the dataset so that:
- row 1 = feature1 (Date)
- row 2 = feature2 (Store)
- row 3 = feature3 (Product)
- row 4 = target (Number_sold)
- row 5 = feature1 (Date)
etc.

We then threw the data into a MinMaxScaler between 1 and 2. Then we split the data into a features array(X) and a target array(y).

As for the algorithm, we used LSTM with 15 hidden units, 3 inputs, and 1 layer.

Important details:
- For the algorithm, the test data and input data is preprocessed with MinMaxScaler. This means the input, output and target are also scaled in the test.py file.
- If you happen to get an error stating that you do not have enough memory, try lowering the hidden_size in model.py. Don't forget to also lower
    the first argument in nn.Linear() as well. In our case, 16 GB of RAM should be enough for hidden_size = 15.