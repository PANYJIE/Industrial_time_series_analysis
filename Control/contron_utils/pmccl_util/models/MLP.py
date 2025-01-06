import torch
from torch import nn


class MLPModel(nn.Module):
    def __init__(self, dropout_prob, input_size, fc1_size, fc2_size, window_size):
        super(MLPModel, self).__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window_size * input_size, fc1_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc1_size, fc2_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(fc2_size, input_size)
        )


    def forward(self, x):

        f1_out = self.fc1(x)

        f2_out = self.fc2(f1_out)

        out = self.fc3(f2_out)
        return out