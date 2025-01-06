from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE import dlutils
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE import memory
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.dlutils import TransformerEncoderLayer, TransformerDecoderLayer, PositionalEncoding
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE import dlutils
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE import constants
torch.manual_seed(1)
MemoryModule = memory.MemoryModule


class MCFMAAE(nn.Module):
    def __init__(self, feats, dataset):
        super(MCFMAAE, self).__init__()
        self.name = 'MCFMAAE'
        self.lr = constants.getlr(dataset)
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.dataset=dataset
        self.pos_encoder = PositionalEncoding(
            2 * feats, 0.1, self.n_window)   # position_encoder
        encoder_layers2 = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder2 = TransformerEncoder(
            encoder_layers2, 1)      # encoder
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(
            decoder_layers2, 1)      # decoder
        self.fcn = nn.Sequential(nn.Linear(
            2 * feats, feats), nn.Linear(feats, feats), nn.Linear(feats, feats), nn.Sigmoid())
        # memory
        self.memory_module = MemoryModule(
            mem_size=512, latent_size=10, hidden_size=2 * feats, batch_size=self.batch)

    def forward(self, src):
        # #Multi scale Convolutional Fusion
        src = torch.cat((src, src), dim=2)
        src = src.transpose(1, 2)
        tcn1=dlutils.gettcn1(self.dataset)
        tcn2=dlutils.gettcn2(self.dataset)
        conv1=dlutils.getconv1(self.dataset)
        conv2=dlutils.getconv2(self.dataset)
        conv3=dlutils.getconv3(self.dataset)
        temp1 = tcn1(src)
        temp2 = tcn2(src)
        src = src.double()
        temp1 = temp1.double()
        temp2 = temp2.double()

        # Multi scale causal expansion convolutional fusion
        src = src + conv1(src) + conv2(src) + \
            conv3(src) + temp1 + temp2
        src = src.transpose(1, 2)

        # position_encoder
        src = src * dlutils.math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        tgt = src

        # encoder
        src = torch.squeeze(src, dim=2)
        src2 = self.transformer_encoder2(src)
        src2 = src2.permute(1, 2, 0)

        # memory
        src2 = self.memory_module(src2)
        src2 = src2.permute(2, 0, 1)

        # decoder
        x2 = self.fcn(self.transformer_decoder2(*(tgt, src2)))

        return x2


class DataDiscriminator(nn.Module):
    def __init__(self, data_size, input_channel, hidden_size):
        super(DataDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.layer_num = int(np.log2(data_size)) - 3
        self.max_channel_num = data_size * 2
        self.final_size = 4
        self.conv_list = []

        for i in range(self.layer_num + 1):
            current_out_channel = self.max_channel_num // 2 ** (
                self.layer_num - i)

            if i == 0:
                self.conv_list.append(nn.Conv1d(in_channels=input_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
            else:
                self.conv_list.append(nn.Conv1d(in_channels=prev_channel, out_channels=current_out_channel,
                                                kernel_size=4, stride=2, padding=1))
                self.conv_list.append(nn.BatchNorm1d(current_out_channel))
            self.conv_list.append(nn.LeakyReLU(0.2, inplace=True))
            prev_channel = current_out_channel

        self.conv_layers = nn.Sequential(*self.conv_list)

        self.linear_layers = nn.Sequential(
            nn.Linear(self.final_size * self.max_channel_num, 1)
        )

    def forward(self, x):
        out = self.conv_layers(x)
        # Calculate the number of elements for the current out
        current_size = out.size(0) * out.size(1) * out.size(2)
        # Calculate the expected number of elements
        expected_size = self.final_size * self.max_channel_num

        # If the current number of elements is greater than the expected number of elements, perform a slicing operation
        if current_size > expected_size:
            # Calculate the number of elements that need to be retained
            remaining_size = expected_size

            k = current_size // expected_size
            out = out.reshape(1, -1)

            out = out[0][:int(k * remaining_size)]

            # Slice tensors based on the number of elements to be retained
            out = out.reshape(-1, self.final_size * self.max_channel_num)

        out = self.linear_layers(out)
        return out
