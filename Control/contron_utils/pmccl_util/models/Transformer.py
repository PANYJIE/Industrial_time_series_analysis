import torch
from torch import nn

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, window_size, num_heads, num_layers, dropout_prob):
        super(TransformerModel, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=self.num_heads),
                                             num_layers=self.num_layers)


        self.linearOut = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * window_size, d_model))


    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.encoder(x)


        out = self.linearOut(x)

        return out