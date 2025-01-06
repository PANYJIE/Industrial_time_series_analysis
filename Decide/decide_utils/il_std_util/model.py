import random
import torch
import torch.nn as nn
from . import xiaobo
from . import DTcov
from math import sqrt
from . import models
from . import TFT
from torch.autograd import Variable
LinearLayer = models.temporal_fusion_t.linear_layer.LinearLayer
AddAndNorm = models.temporal_fusion_t.add_and_norm.AddAndNorm
GLU = models.temporal_fusion_t.gated_linear_unit.GLU
seed = 99
random.seed(seed)
torch.manual_seed(seed)


class FullAttention(nn.Module):
    '''
    The Attention operation
    '''

    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        xiaobo. mWDN_RCF
        DTcov.Dynamic_conv1d
        TFT.TemporalFusionTransformer

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''

    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = FullAttention(
            scale=None, attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, 14)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)


class GatedResidualNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_layer_size,
                 output_size=None,
                 dropout_rate=None,
                 use_time_distributed=True,
                 return_gate=False,
                 batch_first=False
                 ):

        super(GatedResidualNetwork, self).__init__()
        if output_size is None:
            output = hidden_layer_size
        else:
            output = output_size

        self.output = output
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_size = hidden_layer_size
        self.return_gate = return_gate

        self.linear_layer = LinearLayer(
            input_size, output, use_time_distributed, batch_first)

        self.hidden_linear_layer1 = LinearLayer(
            input_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_context_layer = LinearLayer(
            hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)
        self.hidden_linear_layer2 = LinearLayer(
            hidden_layer_size, hidden_layer_size, use_time_distributed, batch_first)

        self.elu1 = nn.ELU()
        self.glu = GLU(hidden_layer_size, output, dropout_rate,
                       use_time_distributed, batch_first)
        self.add_and_norm = AddAndNorm(hidden_layer_size=output)

    def forward(self, x, context=None):
        # Setup skip connection
        if self.output_size is None:
            skip = x
        else:
            skip = self.linear_layer(x)

        # Apply feedforward network
        hidden = self.hidden_linear_layer1(x)
        if context is not None:
            hidden = hidden + self.hidden_context_layer(context)
        hidden = self.elu1(hidden)
        hidden = self.hidden_linear_layer2(hidden)
        gating_layer, gate = self.glu(hidden)
        return gate


class Encoder(nn.Module):
    def __init__(self, p=0.1):
        super(Encoder, self).__init__()
        self.xiaobo = xiaobo.mWDN_RCF(15, 15)
        self.xiaobo1 = xiaobo.mWDN_RCF(15, 15)

        self.self_attention = AttentionLayer(14, 2)
        self.conv = nn.Conv1d(in_channels=14, out_channels=14,
                              kernel_size=3, padding=1, bias=True)
        self.dtcov1 = DTcov. Dynamic_conv1d(14, 14, 1, padding=0)
        self.dtcov2 = DTcov.Dynamic_conv1d(14, 14, 1, padding=0)
        self.fc7 = nn.Linear(840, 64)
        self.fc8 = nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)

        self.vsn1 = TFT.VariableSelectionNetwork(1, 15, 15, 0.05)
        self.vsn3 = TFT.VariableSelectionNetwork(1, 15, 15, 0.05)
        self.vsn7 = TFT.VariableSelectionNetwork(1, 14, 14, 0.05)
        self.vsn8 = TFT.VariableSelectionNetwork(1, 14, 14, 0.05)
        # self.Dconv = DeformableConv1d(in_channels=14,  out_channels=14,kernel_size=3)

    def forward(self, x):
        # x [128,30,21]
        out = x.permute(0, 2, 1)
        # x [128,14,30]
        out_even = out[:, :, ::2]
        out_odd = out[:, :, 1::2]
        a = out_even
        b = out_odd
        a = a.permute(0, 2, 1)
        b = b.permute(0, 2, 1)

        feature = out_even.shape[1]
        Z = []
        for m in range(feature):
            h1, h2, low = self.xiaobo(out_even[:, m, :])
            Z.append(low.unsqueeze(1))

        low = torch.cat(Z, dim=1)
        low = self.conv(low) + low
        low = self.vsn1(low)
        out_even = low
        out_even = torch.cat(
            (out_even, (self.vsn7(b)).permute(0, 2, 1)), dim=2)

        Z = []
        for m in range(feature):
            _, _, low = self.xiaobo1(out_odd[:, m, :])
            Z.append(low.unsqueeze(1))
        low = torch.cat(Z, dim=1)

        low = self.dtcov2(low) + low
        low = self.vsn3(low)

        out_odd = low
        out_odd = torch.cat((out_odd, (self.vsn8(a)).permute(0, 2, 1)), dim=2)

        out = torch.cat((out_even, out_odd), dim=2)

        out = out.reshape(out.size(0), -1)
        out = self.fc7(out)

        out = self.dropout(out)
        out = self.fc8(out)
        return out


if __name__ == '__main__':
    device = 'cuda:1'
    input = torch.ones([128, 30, 21]).to(device)
    encoder = Encoder().to(device)
    out = encoder(input)
