import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(1)
        repeated_hidden = hidden.unsqueeze(1).repeat(1, max_len, 1)
        # print('encoder_outputs', encoder_outputs.shape)
        # print('repeated_hidden', repeated_hidden.shape)
        # print('cat', torch.cat((repeated_hidden, encoder_outputs), dim=2).shape)
        energy = torch.tanh(self.attn(torch.cat((repeated_hidden, encoder_outputs), dim=2)))
        # print('energy', energy.shape)
        attention_scores = self.v(energy).squeeze(2)
        attention_weights = nn.functional.softmax(attention_scores, dim=1)
        context_vector = (encoder_outputs * attention_weights.unsqueeze(2)).sum(dim=1)
        return context_vector, attention_weights
class CNN_GRU_Attention(nn.Module):
    def __init__(self, input_dim, cnn_out_channels, hidden_size_gru, gru_num_layers, attention_hidden_size, output_dim):
        super(CNN_GRU_Attention, self).__init__()
        self.input_dim = input_dim
        self.cnn_out_channels = cnn_out_channels
        self.hidden_size_gru = hidden_size_gru
        self.lstm_num_layers = gru_num_layers
        self.attention_hidden_size = attention_hidden_size
        self.output_dim = output_dim

        # CNN layers
        self.cnn = nn.Sequential(


            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)    # 池化层设计时需要保证输出后的数据时间序列长度不变，池化核和padding要保持一定的计算关系
        )

        self.gru = nn.GRU(input_size=12, hidden_size=hidden_size_gru, num_layers=gru_num_layers,
                          batch_first=True, dropout=0.5)
        self.gru2 = nn.GRU(input_size=hidden_size_gru, hidden_size=256, num_layers=gru_num_layers,
                          batch_first=True, dropout=0.5)

        # Attention mechanism
        self.attention_layer = Attention(attention_hidden_size)
        # Fully connected layer
        self.layer_com1 = nn.Linear(attention_hidden_size, 64)
        self.layer_com2 = nn.Linear(64, output_dim)
    def forward(self, x):
        # CNN
        # print('row_input', x.shape)  # 100*36*14
        # print(x[0, 0, :])
        # adjust dimensions for CNN，我们是在特征维度上进行卷积，CNN中的API默认在最后一个维度上进行卷积计算，因此需要保证输入数据的形状
        # print('cnn_input:', cnn_input.shape)    # 100*14*36
        cnn_output = self.cnn(x)
        # print('cnn_output:', cnn_output[0, :, 0])
        # print('cnn_output:', cnn_output.shape)      # 100*64*36
        #  PyTorch中一维卷积的输入尺寸为：input(batch_size, input_size, seq_len),input_size即为in_channels，
        #  一维卷积中卷积操作是针对最后一个维度，也即seq_len维度进行的，所以如果只想在特征维度上做卷积，需要将in_channels设置为数据中的时序大小
        # 同时为了保证在时序长度上不变，cnn的输出通道大小也要设置为样本数据中的时序窗口大小
        gru_out1, h_n1 = self.gru(cnn_output)
        # print('gru_out1:', gru_out1.shape)  # 100*64*36
        gru_out2, h_n2 = self.gru2(gru_out1)
        # Attention mechanism
        attention_output, attention_weights = self.attention_layer(h_n2[-1], gru_out2)
        # print('attention_output:', attention_output[0])
        # print('attention_output', attention_output.shape)   # 100*64
        # Fully connected layer
        output = F.leaky_relu(self.layer_com1(attention_output))
        output = F.leaky_relu(self.layer_com2(output))
        # print('final_output', output.shape)
        return output