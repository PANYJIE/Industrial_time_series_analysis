import torch
import torch.nn as nn

class LSTMVAEModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMVAEModel, self).__init__()

        self.encoder = RNNModel(input_size=input_size, hidden_size=hidden_size, rnn_type="LSTM")
        self.mean_layer = nn.Linear(hidden_size, hidden_size)
        self.log_var_layer = nn.Linear(hidden_size, hidden_size)
        self.encoder_to_decoder = nn.Linear(hidden_size, input_size)
        self.decoder = Sequence2SequenceDecoder(input_size=hidden_size, hidden_size=input_size, bias=True,
                                                decoder_type="LSTM")

    def forward(self, x):

        b, c, T = x.size()
        x = x.view(-1, T, c)  # (b, c, T) -> (b, T, c)
        _, (hx, _) = self.encoder(x)
        mean = self.mean_layer(hx)
        log_var = self.log_var_layer(hx)
        # reparameter
        z = self.reparameter(mean, log_var)
        z = self.encoder_to_decoder(z.squeeze(0))

        reconstruct_x = self.decoder(x, z)
        reconstruct_x = reconstruct_x.view(-1, c, T)
        return reconstruct_x[:, :, -1]

    def reparameter(self, mean, log_var):
        eps = torch.randn(mean.shape).to(mean.device)
        z = mean + torch.sqrt(torch.exp(log_var)) * eps
        return z

    def encode(self, x):
        # hx: batch_size, hidden_size
        _, (hx, _) = self.encoder(x)
        return hx

    def encode_and_variational(self, x):
        hx = self.encode(x)
        mean = self.mean_layer(hx)
        log_var = self.log_var_layer(hx)
        return mean, log_var

    def decode(self, x, z):
        return self.decoder(x, z)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type="LSTM", bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.bias = bias

        if rnn_type == "LSTM":
            self.rnn_cell = nn.LSTMCell(input_size, hidden_size, bias)
        else:
            raise NotImplementedError

    def forward(self, input, state=None):

        batch_size = input.shape[0]
        sequence_length = input.shape[1]
        if state is not None:
            h, c = state
        else:
            h = torch.zeros(batch_size, self.hidden_size).to(input.device)
            c = torch.zeros(batch_size, self.hidden_size).to(input.device)

        h_list = []
        for i in range(sequence_length):
            h, c = self.rnn_cell(input[:, i, :], (h, c))
            h_list.append(h)
        h = torch.stack(h_list, 1)
        return h, (h_list[-1], c)

    def __repr__(self):
        return "{} (input_size={}, hidden_size={}, bias={})".format(
            self.rnn_type,
            self.input_size,
            self.hidden_size,
            self.bias
        )


class Sequence2SequenceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, decoder_type="LSTM", bias=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.decoder_type = decoder_type
        self.bias = bias

        self.hidden_to_input = nn.Linear(hidden_size, input_size)

        if decoder_type == "LSTM":
            self.decoder = nn.LSTMCell(input_size, hidden_size, bias=bias)
        else:
            raise NotImplementedError

    def forward(self, x, encoder_output_h):

        batch_size, sequence_length, hidden_size = x.shape

        hx = encoder_output_h
        cx = torch.zeros_like(hx).to(encoder_output_h.device)
        input = torch.zeros(batch_size, self.input_size).to(encoder_output_h.device)

        decoder_output = []
        for i in range(sequence_length):
            hx, cx = self.decoder(input, (hx, cx))
            input = self.hidden_to_input(hx.detach())
            decoder_output.append(hx)
        decoder_output = decoder_output[::-1]
        decoder_output = torch.stack(decoder_output, 1)

        return decoder_output

    def __repr__(self):
        return "{} (input_size={}, hidden_size={}, bias={})".format(
            self.decoder_type,
            self.input_size,
            self.hidden_size,
            self.bias
        )