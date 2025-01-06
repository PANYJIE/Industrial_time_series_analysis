import torch
import torch.nn as nn
from torch import Tensor
import torch.fft
import yaml
import h5py
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from matplotlib import pyplot as plt
import numpy as np
import os
import shutil
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
from einops import rearrange, repeat, reduce
from typing import List
from .fno import SpectralConv1d, FNO1d, SpectralConv2d_fast, FNO2d
# # import deepxde as dde
# from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.deepxde.nn.pytorch.fnn import FNN
# from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.deepxde.nn.pytorch.nn import NN
# from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.deepxde.nn import activations
from .oformer import Encoder1D, STDecoder1D, PointWiseDecoder1D



from .anode import ODENet
from .dilated_conv import DilatedConvEncoder


####################################################################
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
            Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

####################################################################
# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t,
                             self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """

    def __init__(self,
                 input_dim,
                 attn_type,                 # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1/64,             # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == inner_dim)
        self.attn_type = attn_type

        self.heads = heads
        self.dim_head = dim_head

        # print("\nINPUT DIM\n")
        self.to_q = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(input_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(input_dim, inner_dim, bias=False)

        if attn_type == 'galerkin':
            self.k_norm = nn.InstanceNorm1d(dim_head)
            self.v_norm = nn.InstanceNorm1d(dim_head)
        elif attn_type == 'fourier':
            self.q_norm = nn.InstanceNorm1d(dim_head)
            self.k_norm = nn.InstanceNorm1d(dim_head)
        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, input_dim),
                # nn.Linear(input_dim, input_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                # nn.Linear(inner_dim + pos_dim*heads, input_dim),
                nn.Linear(inner_dim, input_dim),
                # nn.Linear(input_dim, input_dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / np.sqrt(dim_head)
            self.diagonal_weight = 1. / np.sqrt(dim_head)
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        # if init_params:
        #    self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(
                dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        # for param in self.to_qkv.parameters():
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) *
                            self.dim_head, :], gain=self.init_gain)

                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                        torch.diag(torch.ones(
                            param.size(
                                -1),
                            dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, queries, keys, values, pos=None, not_assoc=False, padding_mask=None):
        if pos is None and self.relative_emb:
            raise Exception(
                'Must pass in coordinates when under relative position embedding mode')

        queries = self.to_q(queries)
        keys = self.to_k(keys)
        values = self.to_k(values)
        queries, keys, values = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), (queries, keys, values))
        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(keys, self.k_norm)
                v = self.norm_wrt_domain(values, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(queries, self.q_norm)
                k = self.norm_wrt_domain(keys, self.k_norm)
        else:
            grid_size = torch.sum(
                padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)  # [b, 1, 1]

            padding_mask = repeat(
                padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]

            # currently only support instance norm
            if self.attn_type == 'galerkin':
                k = rearrange(k, 'b h n d -> (b h) n d')
                v = rearrange(v, 'b h n d -> (b h) n d')

                k = masked_instance_norm(k, padding_mask)
                v = masked_instance_norm(v, padding_mask)

                k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
            else:  # fourier
                q = rearrange(q, 'b h n d -> (b h) n d')
                k = rearrange(k, 'b h n d -> (b h) n d')

                q = masked_instance_norm(q, padding_mask)
                k = masked_instance_norm(k, padding_mask)

                q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(
                padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]

        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], x.device)
                freqs_y = self.emb_module.forward(pos[..., 1], x.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
                k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], x.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=q.shape[1])
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                raise Exception(
                    'Currently doesnt support relative embedding > 2 dimensions')

        elif self.cat_pos:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            if padding_mask is not None:
                padding_mask = ~padding_mask
                padding_mask_arr = torch.matmul(
                    padding_mask, padding_mask.transpose(-1, -2))  # [b, h, n, n]
                mask_value = 0.
                score = score.masked_fill(padding_mask_arr, mask_value)
                out = torch.matmul(score, v) * (1./grid_size)
            else:
                out = torch.matmul(score, v) * (1./q.shape[2])
        else:
            if padding_mask is not None:
                q = q.masked_fill(~padding_mask, 0)
                k = k.masked_fill(~padding_mask, 0)
                v = v.masked_fill(~padding_mask, 0)
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1. / grid_size)
            else:
                # print(k.shape)
                # print(v.shape)
                # print(queries.shape)
                dots = torch.matmul(keys.transpose(-1, -2), values)
                out = torch.matmul(queries, dots) * (1./queries.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(out.shape)
        # print(self.to_out)
        # return self.to_out(out), None
        # return self.to_out(out), dots
        # return self.to_out(out), dots
        # print(self.to_k.weight)
        return self.to_out(out), self.to_q.weight

####################################################################

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x)


class MLPNet(nn.Module):
    """
    """

    def __init__(self, data_dim, hidden_dim):
        super(MLPNet, self).__init__()
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x)

####################################################################
# MANO
####################################################################


class MANO1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, augment_dim, neural_operator, kernels: List[int], rep_dim=200, depth=10, time_dependent=True, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dependent = time_dependent
        self.augment_dim = augment_dim

        component_dims = rep_dim // 2
        self.component_dims = component_dims

        # Get input processing
        self.vhh_embedding_layer = nn.Linear(1, hidden_dim, bias=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Encoding/embedding
        self.embedding = torch.nn.Embedding(500, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        self.Feature_Projection_layers = nn.Linear(500, 100, bias=True)

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        # self.input_fc1 = nn.Linear(input_dim, hidden_dim)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dim,
            [hidden_dim] * depth + [rep_dim],
            kernel_size=3
        )

        # self.feature_extractor1 = DilatedConvEncoder(
        #     hidden_dim,
        #     [hidden_dim] * depth + [rep_dim],
        #     kernel_size=3
        # )

        self.kernels = kernels
        self.multi_cnn = nn.ModuleList(
            [nn.Conv1d(rep_dim, component_dims, k, padding=k-1)
             for k in kernels]
        )

        # self.multi_cnn1 = nn.ModuleList(
        #     [nn.Conv1d(rep_dim, component_dims, k, padding=k-1) for k in kernels]
        # )

        self.LinearAttention = LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
                                               dim_head=hidden_dim, dropout=0.1,
                                               relative_emb=False,
                                               # relative_emb=True,
                                               init_method='xavier',
                                               init_gain=1.
                                               )
        self.project = nn.Linear(hidden_dim, 10)

        self.q_h_project = nn.Linear(hidden_dim, 10)

        self.ODENet = ODENet(input_dim, hidden_dim, augment_dim,
                             self.time_dependent)

        self.Norm = nn.LayerNorm(hidden_dim, 100)

        # Internal Physics Model
        self.neural_operator = neural_operator
        # Layers and dropout
        self.dropout = nn.Dropout(dropout)

        # Output decoding layer
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.act = nn.SiLU()

    def forward(self, queries, keys, values, t):  # , mask):

        x = torch.swapaxes(values.clone(), 1, 2)
        out1 = values[:, -1, :].squeeze()
        # print(out1.shape)
        res_x = x.clone()  # [128, 100, 10]
        # [128, 1000] --> [128,1000]
        res_x = res_x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        xf = torch.fft.fft(res_x, dim=1).to(torch.float32)


################################################################
        # Embed query
        fc_x = self.input_fc(x)
        # conv encoder
        # [batchsize,hidden_dim,seqlength]   [128,100,100] 中间100是隐示表征，后100是seqlength
        fc_x = fc_x.transpose(1, 2)
        mv1 = self.feature_extractor(fc_x)

        multi_hidden = []
        for idx, mod in enumerate(self.multi_cnn):
            # b d t   #[128,100,100]  101,103,107,    [batchsize,hidden_dim,seqlength]
            out = mod(mv1)
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            multi_hidden.append(out.transpose(1, 2))  # b t d
        multi_hidden = reduce(
            rearrange(multi_hidden, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )  # [128,100,100]     [batchsize,seqlength,hidden_dim]
        q_h = multi_hidden.clone()
        # print("656 q_h shape is {}".format(q_h.shape))


################################################################
        # Embed Values   FFT
        v = xf.view(x.shape[0], x.shape[1], x.shape[2])
        fc_v = self.input_fc(v)
        # conv encoder
        # [batchsize,hidden_dim,seqlength]   [128,100,100] 中间100是隐示表征，后100是seqlength
        fc_v = fc_v.transpose(1, 2)
        mv2 = self.feature_extractor(fc_v)

        multi_hidden_v = []
        for idx, mod in enumerate(self.multi_cnn):
            # b d t   #[128,100,100]  101,103,107,    [batchsize,hidden_dim,seqlength]
            out_v = mod(mv2)
            if self.kernels[idx] != 1:
                out_v = out_v[..., :-(self.kernels[idx] - 1)]
            multi_hidden_v.append(out_v.transpose(1, 2))  # b t d
        multi_hidden_v = reduce(
            rearrange(multi_hidden_v, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )  # [128,100,100]     [batchsize,seqlength,hidden_dim]

        v_h = multi_hidden_v.clone()
        # print("680 v_h shape is {}".format(v_h.shape))

        # Embed keys
        # # static data Feature Projection
        # ReLU = torch.nn.ReLU()
        keys = self.embedding(keys.long()) * np.sqrt(self.hidden_dim)
        keys = self.pos_encoding(keys)  # [128,500,64]
        keys = self.dropout(keys)
        # # Embed learned tokens
        k_h = torch.swapaxes(self.Feature_Projection_layers(
            torch.swapaxes(keys, 1, 2)), 1, 2)  # [128,100,100]
        # print("689 k_h shape is {}".format(k_h.shape))

        # attention fusion
        # # SANO
        x_h1, _ = self.LinearAttention(q_h, k_h, k_h)  #
        x_h2, _ = self.LinearAttention(q_h, v_h, v_h)  #
        x_h = 0.5 * x_h1 + 0.5 * x_h2

        # # ablation w/o FFT
        # x_h, _ = self.LinearAttention(q_h, k_h, k_h)  #

        # ablation w/o Equation
        # x_h, _ = self.LinearAttention(q_h, v_h, v_h)  #

        # # ablation w/o fft equation
        # x_h, _ = self.LinearAttention(q_h, q_h, q_h)  #

        x_h = self.project(x_h)

        # [128,100] src没有squeeze
        x1 = self.neural_operator(x_h, queries.unsqueeze(-1))[..., 0].squeeze()
        if (len(x1.shape) == 2):
            x1 = x1.unsqueeze(-1)

        res_q = self.q_h_project(q_h).transpose(1, 2)
        dx = x1 + res_q[:, -1, :].unsqueeze(-1)  # [128,100,1]

        # Embed Values
        v_hh = self.vhh_embedding_layer(dx)  # [128,100,64] # [128, 100, 100]

        # v_hh1 = v_hh.clone()
        # ## 无注意力 无eq_embedding
        feats = self.ODENet(v_hh, True)

        # 有注意力
        # feats = self.ODENet(k_h, v_h, True)
        # print("feats shape is {} ".format(feats.shape))
        # norm ?
        out0 = self.output_layers(feats)[..., 0]
        # return x[...,0] + out
        preds1 = x1[..., 0] + out0 + out1
        return preds1


