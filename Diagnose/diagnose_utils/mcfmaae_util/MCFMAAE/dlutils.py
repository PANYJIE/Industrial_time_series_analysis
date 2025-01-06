import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.tcn import TemporalConvNet
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.parser import *

feadim = args.dataset
label = {"MBA": 4, "MSL": 110, "NAB": 2, "SMAP": 50, "SMD": 76, "SWaT": 2, "UCR": 2, "WADI": 254}


def gettcn1(dataset):
    feadim = label[dataset]
    tcn1 = TemporalConvNet(num_inputs=feadim, num_channels=[feadim, feadim, feadim, feadim], kernel_size=3, dropout=0.2)
    return tcn1
def gettcn2(dataset):
    feadim = label[dataset]
    tcn2 = TemporalConvNet(num_inputs=feadim, num_channels=[feadim, feadim, feadim, feadim], kernel_size=5, dropout=0.2)
    return tcn2
def getconv1(dataset):
    feadim = label[dataset]
    conv1 = nn.Conv1d(in_channels=feadim, out_channels=feadim, kernel_size=1, padding=0).double()
    return conv1
def getconv2(dataset):
    feadim = label[dataset]
    conv2 = nn.Conv1d(in_channels=feadim, out_channels=feadim, kernel_size=3, padding=1).double()
    return conv2
def getconv3(dataset):
    feadim = label[dataset]
    conv3 = nn.Conv1d(in_channels=feadim, out_channels=feadim, kernel_size=5, padding=2).double()
    return conv3

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, src,src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
