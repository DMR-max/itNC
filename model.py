import torch.nn as nn

class RecurrentNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=15, num_layers=1, batch_first=True)
        self.linear = nn.Linear(15, 1)

    def forward(self, x):
        # H_0 and C_0 not needed as the model can figure it out
        x, _ = self.lstm(x)
        x = self.linear(x)
        
        return x