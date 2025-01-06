import torch
from torch import nn


class LSTModel(nn.Module):
    def __init__(self, dropout_prob, input_size, hidden_size, window_size, num_layers):
        super(LSTModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout = nn.Dropout(dropout_prob)
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=self.num_layers)

        self.linearOut = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * window_size, input_size))

    def forward(self, x):
        print("调用了LSTM")
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = torch.tanh(torch.transpose(x, 1, 2))
        x = x.permute(0, 2, 1)

        out = self.linearOut(x)

        return out