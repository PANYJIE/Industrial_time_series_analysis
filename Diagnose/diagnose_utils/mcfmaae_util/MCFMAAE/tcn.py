import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.tensorboard import SummaryWriter


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        In fact, this is a cropping module that cuts out extra padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        Equivalent to a Residual block

        :param n_inputs: int, number of input channels
        :param n_outputs: int, number of output channels
        :param kernel_size: int, size of the convolution kernel
        :param stride: int, stride, usually 1
        :param dilation: int, dilation factor
        :param padding: int, padding factor
        :param dropout: float, dropout rate
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # After conv1, the output size is actually (Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # Crop out the extra padding part to maintain the output timestep as seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # Crop out the extra padding part to maintain the output timestep as seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialize the parameters
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        x = torch.tensor(x, dtype=torch.float)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    # A dilated causal convolution
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN, the TCN structure currently provided in the paper supports the case where each time step is a number,
        i.e., a sequence structure. For the case where each time step is a vector, the vector can be divided into
        several input channels for that time step. However, for the case where each time step is a matrix or a higher
        dimensional image, it is not easy to handle.

        :param num_inputs: int, number of input channels
        :param num_channels: list, number of hidden channels for each layer, e.g., [25,25,25,25] means 4 hidden layers
        with each layer having 25 hidden channels
        :param kernel_size: int, size of the convolution kernel
        :param dropout: float, dropout rate
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # Dilation factor: 1, 2, 4, 8...
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # Determine the input channels for each layer
            out_channels = num_channels[i]  # Determine the output channels for each layer
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        The input structure of x is different from RNN. Generally, the size of RNN is (Batch, seq_len, channels) or
        (seq_len, Batch, channels). Here, the seq_len is placed after the channels, and all time step data is
        concatenated as the input size of Conv1d, realizing convolution across time steps, a very clever design.

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)