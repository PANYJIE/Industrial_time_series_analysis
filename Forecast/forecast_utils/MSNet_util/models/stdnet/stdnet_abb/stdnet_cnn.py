# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/models_nbeats__nbeats.ipynb (unless otherwise specified).

__all__ = ['IdentityBasis', 'init_weights', 'ACTIVATIONS', 'STDNet']

# Cell

from functools import partial
from typing import List, Tuple

import numpy as np
import torch as t
import torch.nn as nn

from .components_ab_cnn import TimeCNNBasis, SpaceCNNBasis, TemporalCNNATTBasis, SpacialCNNATTBasis, TimeCNNStartBasis, \
    SpaceCNNStartBasis
from .components_att import TemporalAutoBasis


class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


# Cell
class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


# Cell
class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta):
        backcast = theta[:, :, :self.backcast_size]
        forecast = theta[:, :, -self.forecast_size:]
        return backcast, forecast


# Cell
def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == 'orthogonal':
            t.nn.init.orthogonal_(module.weight)
        elif initialization == 'he_uniform':
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == 'he_normal':
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == 'glorot_uniform':
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == 'glorot_normal':
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == 'lecun_normal':
            pass  # t.nn.init.normal_(module.weight, 0.0, std=1/np.sqrt(module.weight.numel()))
        else:
            assert 1 < 0, f'Initialization {initialization} not found'


# Cell
ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


import random


class Model(nn.Module):
    def __init__(self,
                 configs=None,
                 # n_time_in: int = 96,
                 # n_time_out: int = 96,
                 # n_x: int = 0,
                 # n_s: int = 0,
                 # n_dim: int = 0,
                 # shared_weights: bool = False,
                 # activation: str = 'ReLU',
                 initialization: str = 'lecun_normal',
                 # stack_types: List[str] = ['time_auto', 'space_cnn_att', 'space_cnn_att'],
                 # rate1: int = 0.5,
                 # start_stack_types: List[str] = 3 * ['None'],
                 # n_cnn_kernel_size: List[int] = [3, 3, 3],
                 # n_blocks: List[int] = [3, 3, 3],
                 # n_layers: List[int] = 3 * [2],
                 # n_mlp_units: List[List[int]] = 3 * [[256, 256]],
                 # cnnatt_hidden_dim: int = 256,
                 # n_harmonics: int = 5,
                 # n_polynomials: int = 5,
                 n_x_hidden=None,
                 n_s_hidden=None,
                 # batch_normalization: bool = False,
                 # dropout_prob_theta: float = 0.,
                 random_seed: int = 1

                 ):
        super(Model, self).__init__()







        if n_s_hidden is None:
            n_s_hidden = [0]
        if n_x_hidden is None:
            n_x_hidden = [0]
        # self.save_hyperparameters()
        if configs.activation == 'relu':
            activation = 'ReLU'

        if activation == 'SELU': initialization = 'lecun_normal'

        # ------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.n_time_in = configs.seq_len
        self.n_time_out = configs.pred_len
        self.n_x = configs.n_x
        self.n_x_hidden = n_x_hidden
        self.n_s = configs.n_s
        self.n_s_hidden = n_s_hidden
        self.n_dim = configs.c_out
        self.shared_weights = configs.shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = configs.stack_types
        self.rate1 = configs.rate1
        self.start_stack_types = configs.start_stack_types
        self.n_cnn_kernel_size = configs.n_cnn_kernel_size
        self.n_blocks = configs.n_blocks
        self.n_layers = configs.n_layers
        self.n_harmonics = configs.n_harmonics
        self.n_polynomials = configs.n_polynomials
        self.n_mlp_units = configs.n_mlp_units
        self.cnnatt_hidden_dim = configs.cnnatt_hidden_dim

        # Regularization and optimization parameters
        self.batch_normalization = configs.batch_normalization
        self.dropout_prob_theta = configs.dropout
        self.random_seed = random_seed

        # Data parameters
        self.return_decomposition = False

        self.model = _STDNet(n_time_in=self.n_time_in,
                             n_time_out=self.n_time_out,
                             n_s=self.n_s,
                             n_x=self.n_x,
                             n_dim=self.n_dim,
                             n_s_hidden=self.n_s_hidden,
                             n_x_hidden=self.n_x_hidden,
                             n_polynomials=self.n_polynomials,
                             n_harmonics=self.n_harmonics,
                             stack_types=self.stack_types,
                             rate1=self.rate1,
                             start_stack_types=self.start_stack_types,
                             n_cnn_kernel_size=self.n_cnn_kernel_size,
                             n_blocks=self.n_blocks,
                             n_layers=self.n_layers,
                             n_mlp_units=self.n_mlp_units,
                             cnnatt_hidden_dim=self.cnnatt_hidden_dim,
                             dropout_prob_theta=self.dropout_prob_theta,
                             activation=self.activation,
                             initialization=self.initialization,
                             batch_normalization=self.batch_normalization,
                             shared_weights=self.shared_weights)

    def on_fit_start(self):
        t.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def forward(self, X, batch_x_mark, dec_inp, batch_y_mark):
        X = X.permute(0, 2, 1)

        if self.return_decomposition:
            outsample_y, forecast, block_forecast, outsample_mask = self.model(X=X,
                                                                               return_decomposition=True)
            return outsample_y, forecast, block_forecast, outsample_mask

        forecast = self.model(X=X, return_decomposition=False).permute(0, 2, 1)
        return forecast


class _STDNetBlock(nn.Module):

    def __init__(self, n_time_in: int, n_time_out: int, n_x: int,
                 n_s: int, n_s_hidden: int, n_dim: int, n_theta: int, n_mlp_units: list,
                 start_basis: nn.Module,
                 end_basis: nn.Module,
                 n_layers: int, batch_normalization: bool, dropout_prob: float, activation: str):
        """
        """
        super().__init__()

        if n_s == 0:
            n_s_hidden = 0
        n_mlp_units = [n_time_in + (n_time_in + n_time_out) * n_x + n_s_hidden] + n_mlp_units

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.n_x = n_x
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ = getattr(nn, activation)()

        hidden_layers = []
        for i in range(n_layers):

            hidden_layers.append(nn.Linear(n_mlp_units[i], n_mlp_units[i + 1]))

            hidden_layers.append(activ)

            # hidden_layers.append(SelectItem(0))
            if self.batch_normalization:
                # hidden_layers.append(nn.BatchNorm1d(num_features=n_mlp_units[i+1]))
                hidden_layers.append(nn.BatchNorm1d(num_features=n_dim))
            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))
        output_layer = [nn.Linear(n_mlp_units[-1], n_theta)]

        if str(end_basis).startswith('Identity'):
            layers = hidden_layers + output_layer
        else:
            layers = hidden_layers

        # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(in_features=n_s, out_features=n_s_hidden)
        self.layers2 = layers
        self.layers = nn.Sequential(*layers)
        self.end_basis = end_basis
        self.start_basis = start_basis

    def forward(self, insample_y):

        batch_size = len(insample_y)

        theta = insample_y
        h0 = None
        if self.start_basis != None:
            theta = theta + self.start_basis(theta)
        for i, net in enumerate(self.layers2):
            theta = net(theta)
        backcast, forecast = self.end_basis(theta)

        return backcast, forecast


# Cell
class _STDNet(nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self,
                 n_time_in,
                 n_time_out,
                 n_s,
                 n_x,
                 n_s_hidden,
                 n_x_hidden,
                 n_dim,
                 n_polynomials,
                 n_harmonics,
                 stack_types: list,
                 rate1: int,
                 start_stack_types: list,
                 n_cnn_kernel_size: list,
                 n_blocks: list,
                 n_layers: list,
                 n_mlp_units: list,
                 cnnatt_hidden_dim: int,
                 dropout_prob_theta,
                 activation,
                 initialization,
                 batch_normalization,
                 shared_weights):
        super().__init__()

        self.n_time_out = n_time_out

        blocks = self.create_stack(stack_types=stack_types,
                                   rate1=rate1,
                                   n_blocks=n_blocks,
                                   n_time_in=n_time_in,
                                   n_time_out=n_time_out,
                                   n_x=n_x,
                                   n_x_hidden=n_x_hidden,
                                   n_s=n_s,
                                   n_s_hidden=n_s_hidden,
                                   n_dim=n_dim,
                                   n_layers=n_layers,
                                   n_mlp_units=n_mlp_units,
                                   cnnatt_hidden_dim=cnnatt_hidden_dim,
                                   batch_normalization=batch_normalization,
                                   dropout_prob_theta=dropout_prob_theta,
                                   activation=activation,
                                   shared_weights=shared_weights,
                                   n_polynomials=n_polynomials,
                                   n_harmonics=n_harmonics,
                                   initialization=initialization,
                                   start_stack_types=start_stack_types,
                                   n_cnn_kernel_size=n_cnn_kernel_size, )
        self.blocks = t.nn.ModuleList(blocks)

    def create_stack(self, stack_types, start_stack_types, n_blocks, n_cnn_kernel_size, rate1, cnnatt_hidden_dim,
                     n_time_in, n_time_out,
                     n_x, n_x_hidden, n_s, n_s_hidden, n_dim,
                     n_layers, n_mlp_units, batch_normalization, dropout_prob_theta,
                     activation, shared_weights,
                     n_polynomials, n_harmonics, initialization):

        block_list = []
        print(n_mlp_units[0][0])
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):

                # Batch norm only on first block
                if (len(block_list) == 0) and (batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                if start_stack_types[i] == 'None':
                    start_basis = None
                elif start_stack_types[i] == 'space_cnn':
                    start_basis = SpaceCNNStartBasis(n_time_in=n_time_in, n_time_out=n_time_out, n_dim=n_dim,
                                                     old_hidden_dim=n_mlp_units[0][0], kernel_size=n_cnn_kernel_size[0])
                elif start_stack_types[i] == 'time_cnn':
                    start_basis = TimeCNNStartBasis(n_time_in=n_time_in, n_time_out=n_time_out, n_dim=n_dim,
                                                    old_hidden_dim=n_mlp_units[0][0], kernel_size=n_cnn_kernel_size[0])
                # elif start_stack_types[i] == 'space_att':
                #     start_basis = StartSpaceATTBasis(n_time_in=n_time_in, n_time_out=n_time_out, n_dim=n_dim)
                # elif start_stack_types[i] == 'time_att':
                #     start_basis = StartTimeATTBasis(n_time_in=n_time_in, n_time_out=n_time_out, n_dim=n_dim)

                # Shared weights
                if shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:

                    if stack_types[i] == 'identity':
                        n_theta = n_time_in + n_time_out
                        end_basis = IdentityBasis(backcast_size=n_time_in,
                                                  forecast_size=n_time_out)


                    elif stack_types[i] == 'space_cnn':
                        n_theta = n_time_in + n_time_out
                        end_basis = SpaceCNNBasis(n_time_in, n_time_out, n_dim, n_cnn_kernel_size[block_id])

                    elif stack_types[i] == 'time_cnn':
                        n_theta = n_time_in + n_time_out
                        end_basis = TimeCNNBasis(n_time_in, n_time_out, n_dim, n_cnn_kernel_size[block_id])

                    elif stack_types[i] == 'space_cnn_att':
                        n_theta = n_time_in + n_time_out
                        end_basis = SpacialCNNATTBasis(n_time_in, n_time_out, n_dim,
                                                       kernel_conv=n_cnn_kernel_size[block_id], rate1=rate1,
                                                       old_hidden_dim=n_mlp_units[0][0], hidden_dim=cnnatt_hidden_dim,
                                                       drop=dropout_prob_theta)

                    elif stack_types[i] == 'time_cnn_att':
                        n_theta = n_time_in + n_time_out
                        end_basis = TemporalCNNATTBasis(n_time_in, n_time_out, n_dim,
                                                        kernel_conv=n_cnn_kernel_size[block_id], rate1=rate1,
                                                        old_hidden_dim=n_mlp_units[0][0], hidden_dim=cnnatt_hidden_dim,
                                                        drop=dropout_prob_theta)

                    elif stack_types[i] == 'time_auto':
                        n_theta = n_time_in + n_time_out
                        end_basis = TemporalAutoBasis(n_time_in, n_time_out, n_dim, drop=dropout_prob_theta,
                                                      hidden_dim=cnnatt_hidden_dim)

                    else:
                        assert 1 < 0, f'Block type not found!'

                    nbeats_block = _STDNetBlock(n_time_in=n_time_in,
                                                n_time_out=n_time_out,
                                                n_x=n_x,
                                                n_s=n_s,
                                                n_dim=n_dim,
                                                n_s_hidden=n_s_hidden,
                                                n_theta=n_theta,
                                                n_mlp_units=n_mlp_units[i],
                                                end_basis=end_basis,
                                                start_basis=start_basis,
                                                n_layers=n_layers[i],
                                                batch_normalization=batch_normalization_block,
                                                dropout_prob=dropout_prob_theta,
                                                activation=activation)

                # Select type of evaluation and apply it to all layers of block
                init_function = partial(init_weights, initialization=initialization)
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, X, return_decomposition: bool = False):

        # insample
        insample_y = X

        # if return_decomposition:
        #     forecast, block_forecasts = self.forecast_decomposition(insample_y=insample_y,
        #                                                             insample_x_t=insample_x_t,
        #                                                             insample_mask=insample_mask,
        #                                                             outsample_x_t=outsample_x_t,
        #                                                             x_s=S)
        #     return outsample_y, forecast, block_forecasts, outsample_mask

        # else:
        forecast = self.forecast(insample_y=insample_y)
        return forecast

    def forecast(self, insample_y):

        residuals = insample_y.flip(dims=(-1,))

        forecast = insample_y[:, :, -1:]  # Level with Naive1
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        return forecast

    def forecast_decomposition(self, insample_y: t.Tensor, insample_x_t: t.Tensor, insample_mask: t.Tensor,
                               outsample_x_t: t.Tensor, x_s: t.Tensor):

        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        n_batch, n_channels, n_t = outsample_x_t.size(0), outsample_x_t.size(1), outsample_x_t.size(2)

        level = insample_y[:, -1:]  # Level with Naive1
        block_forecasts = [level.repeat(1, n_t)]

        forecast = level
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, insample_x_t=insample_x_t,
                                             outsample_x_t=outsample_x_t, x_s=x_s)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_t)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        return forecast, block_forecasts


if __name__ == '__main__':
    model = Model(96, 96)
    input = t.rand(32, 7, 96)
    output = model(input)
    print(output)
