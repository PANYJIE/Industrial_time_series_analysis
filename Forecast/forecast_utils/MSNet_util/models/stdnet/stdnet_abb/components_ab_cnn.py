import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

    def __repr__(self):
        if self.contiguous:
            return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else:
            return f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"


def init_rate(tensor, num=0.5):
    if tensor is not None:
        tensor.data.fill_(num)


class SpaceCNNStartBasis(nn.Module):
    def __init__(self, n_time_in, n_time_out, n_dim, old_hidden_dim=256, kernel_size=2, drop=0):
        super(SpaceCNNStartBasis, self).__init__()

        self.conv = nn.Sequential(*[
            Transpose(1, 2),
            nn.Conv1d(in_channels=n_time_in, out_channels=n_time_in, kernel_size=(kernel_size,),
                      stride=(1,), padding='same'),
            Transpose(1, 2),
            #            nn.Linear(n_time_in, n_time_in)
        ])

    def forward(self, x, insample_x_t=None, outsample_x_tt=None):
        x = self.conv(x)
        return x


class TimeCNNStartBasis(nn.Module):
    def __init__(self, n_time_in, n_time_out, n_dim, old_hidden_dim=256, kernel_size=2, drop=0):
        super(TimeCNNStartBasis, self).__init__()

        self.conv = nn.Sequential(*[
            nn.Conv1d(in_channels=n_dim, out_channels=n_dim, kernel_size=(kernel_size,),
                      stride=(1,), padding='same'),
            #            nn.Linear(n_time_in, n_time_in)
        ])

    def forward(self, x, insample_x_t=None, outsample_x_tt=None):
        x = self.conv(x)
        return x


class SpaceCNNBasis(nn.Module):
    def __init__(self, n_time_in, n_time_out, n_dim, kernel_size=2, drop=0):
        super(SpaceCNNBasis, self).__init__()

        self.conv = nn.Sequential(*[
            Transpose(1, 2),
            nn.Conv1d(in_channels=n_time_in, out_channels=n_time_in, kernel_size=(kernel_size,),
                      stride=(1,), padding='same'),
            Transpose(1, 2),
            nn.Dropout(drop),
            nn.Linear(n_time_in, n_time_in + n_time_out)
        ])
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out

    def forward(self, x, insample_x_t=None, outsample_x_tt=None):
        x = self.conv(x)
        backcast = x[:, :, :self.n_time_in]
        forecast = x[:, :, -self.n_time_out:]
        return backcast, forecast


class TimeCNNBasis(nn.Module):
    def __init__(self, n_time_in, n_time_out, n_dim, kernel_size=2, drop=0):
        super(TimeCNNBasis, self).__init__()
        self.conv = nn.Sequential(*[
            nn.Conv1d(in_channels=n_dim, out_channels=n_dim, kernel_size=(kernel_size,), stride=(1,), padding='same'),
            nn.Dropout(drop),
            nn.Linear(n_time_in, n_time_in + n_time_out)
        ])
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out

    def forward(self, x, insample_x_tt=None, outsample_x_tt=None):
        x = self.conv(x)
        backcast = x[:, :, :self.n_time_in]
        forecast = x[:, :, -self.n_time_out:]
        return backcast, forecast


class TemporalCNNATTBasis(nn.Module):
    def __init__(self, dim_in, dim_out, n_dim, n_kernels=288, old_hidden_dim=256, hidden_dim=256, n_head=8, bias=False,
                 drop=0.1,
                 kernel_conv=3, rate1=0):
        super(TemporalCNNATTBasis, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.kernel_conv = kernel_conv
        self.head_dim = hidden_dim // n_head
        self.conv_q = nn.Conv1d(in_channels=dim_in, out_channels=dim_in, kernel_size=(1,), padding=0, bias=bias)
        self.conv_k = nn.Conv1d(in_channels=dim_in, out_channels=dim_in, kernel_size=(1,), padding=0, bias=bias)
        self.conv_v = nn.Conv1d(in_channels=dim_in, out_channels=dim_in, kernel_size=(1,), padding=0, bias=bias)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head, dropout=drop, batch_first=True)
        self.encoder = nn.Linear(n_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, n_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.predict_linear = nn.Linear(dim_in, dim_in + dim_out)
        self.re = nn.Linear(old_hidden_dim, dim_in)

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU()

        self.conv = TimeCNNBasis(dim_in, dim_out, n_dim, kernel_size=kernel_conv, drop=drop)

        self.reset_parameters(rate1)

    def reset_parameters(self, rate1=1):
        init_rate(self.rate1, num=rate1)
        init_rate(self.rate2, num=1 - rate1)

    def predict(self, x):
        x = self.decoder(x).transpose(1, 2)
        x = self.predict_linear(x)
        backcast = x[:, :, :self.dim_in]
        forecast = x[:, :, -self.dim_out:]
        return backcast, forecast

    def forward(self, x, insample_x_t=None, outsample_x_t=None):
        #        print("rate1 {}, rate2{}".format(self.rate1, self.rate2))
        x = self.re(x)
        conv_in = x

        # att

        x = x.transpose(1, 2)
        x = self.encoder(x)
        att_out = x
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        x = self.attn(q, k, v)[0]
        att_out = self.norm(att_out + x)
        att_backcast, att_forecast = self.predict(att_out)

        # conv
        conv_backcast, conv_forecast = self.conv(conv_in)

        backcast = self.rate1 * conv_backcast + self.rate2 * att_backcast
        forecast = self.rate1 * conv_forecast + self.rate2 * att_forecast

        return backcast, forecast


#         return conv_backcast, conv_forecast

class SpacialCNNATTBasis(nn.Module):
    def __init__(self, dim_in, dim_out, n_dim, n_kernels=288, old_hidden_dim=256, hidden_dim=256, n_head=8, bias=False,
                 drop=0.1,
                 kernel_conv=3, rate1=0):
        super(SpacialCNNATTBasis, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.n_head = n_head
        self.hidden_dim = hidden_dim
        self.kernel_conv = kernel_conv
        self.head_dim = hidden_dim // n_head
        self.conv_q = nn.Conv1d(in_channels=n_dim, out_channels=n_dim, kernel_size=(1,), padding=0, bias=bias)
        self.conv_k = nn.Conv1d(in_channels=n_dim, out_channels=n_dim, kernel_size=(1,), padding=0, bias=bias)
        self.conv_v = nn.Conv1d(in_channels=n_dim, out_channels=n_dim, kernel_size=(1,), padding=0, bias=bias)
        self.attn = nn.MultiheadAttention(hidden_dim, n_head, dropout=drop, batch_first=True)
        self.encoder = nn.Linear(dim_in, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, dim_in)
        self.norm = nn.LayerNorm(hidden_dim)
        self.predict_linear = nn.Linear(dim_in, dim_in + dim_out)
        self.re = nn.Linear(old_hidden_dim, dim_in)

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.relu = nn.ReLU()

        self.conv = SpaceCNNBasis(dim_in, dim_out, n_dim, kernel_size=kernel_conv, drop=drop)

        self.reset_parameters(rate1)

    def reset_parameters(self, rate1=1):
        init_rate(self.rate1, num=rate1)
        init_rate(self.rate2, num=1 - rate1)

    def predict(self, x):
        x = self.decoder(x)
        x = self.predict_linear(x)
        backcast = x[:, :, :self.dim_in]
        forecast = x[:, :, -self.dim_out:]

        return backcast, forecast

    def forward(self, x, insample_x_t=None, outsample_x_t=None):
        x = self.re(x)
        conv_in = x
        # att

        x = self.encoder(x)
        att_out = x
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        x = self.attn(q, k, v)[0]
        att_out = self.norm(att_out + x)
        att_backcast, att_forecast = self.predict(att_out)

        # conv
        conv_backcast, conv_forecast = self.conv(conv_in)

        backcast = 0 * conv_backcast + self.rate2 * att_backcast
        forecast = 0 * conv_forecast + self.rate2 * att_forecast

        return backcast, forecast


if __name__ == '__main__':
    model = TemporalCNNATTBasis(dim_in=288, dim_out=96, n_dim=8)
    input = torch.rand(64, 8, 256)
    output = model(input)
    print(output)

    model = SpacialCNNATTBasis(dim_in=288, dim_out=96, n_dim=8)
    input = torch.rand(64, 8, 256)
    output = model(input)
    print(output)

    model = SpaceCNNStartBasis(n_time_in=96, n_time_out=96, n_dim=8)
    input = torch.rand(64, 8, 96)
    output = model(input)
    print(output)

    model = TimeCNNStartBasis(n_time_in=96, n_time_out=96, n_dim=8)
    input = torch.rand(64, 8, 96)
    output = model(input)
    print(output)
