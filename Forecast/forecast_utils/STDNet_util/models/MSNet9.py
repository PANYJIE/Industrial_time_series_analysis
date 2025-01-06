from typing import Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from Industrial_time_series_analysis.Forecast.forecast_utils.STDNet_util.layers.interpolate import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

# ------------------------ Cubic Spline Interpolation ------------------------ #
def cubic_spline(x, size):
    L = x.size()[1]
    t = torch.linspace(0, 1, L).cuda()
    want = torch.linspace(0, 1, size).cuda()
    coeffs = natural_cubic_spline_coeffs(t, x)
    spline = NaturalCubicSpline(coeffs)
    out = spline.evaluate(want)
    return out

# -------------- Multi Resolution Convolution Extraction Module -------------- #
class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=window_size,
                                  stride=window_size)
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int, out_features: int = 1):
        super().__init__()
        self.out_features = out_features
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, :, : self.backcast_size]
        forecast = theta[:, :, self.backcast_size:]
        return backcast, forecast




class Model(nn.Module):

    def __init__(
            self,
            configs,
            individual=False,

    ):
        # Inherit BaseWindows class
        super(Model, self).__init__()
        self.task_name = configs.task_name

        self.h = configs.pred_len
        self.configs = configs

        self.decompose_forecast = False

        stack_types = ["identity", "identity", "identity"]
        n_blocks = [1, 1, 1]
        mlp_units = 3 * [[configs.d_model, configs.d_model]]
        dropout_prob_theta = configs.dropout
        activation = "ReLU"
        shared_weights = False
        n_harmonics = 2
        n_polynomials = 2

        conv_size_one = configs.conv_size_one

        # conv_size = [18, 4] # 0.333
        # 常用！conv_size = [12, 4] # 0.334


        conv_size = [12, 4] #

        # self.encoder = nn.Linear(configs.seq_len, configs.d_model)
        # self.prediction = nn.Sequential(
        #     nn.Linear(configs.d_model, configs.d_model // 2),
        #     nn.ReLU(),
        #     nn.Linear(configs.d_model // 2, configs.pred_len)
        # )
        self.encoder = nn.Linear(configs.seq_len, configs.pred_len)
        # self.encoder = nn.Sequential(
        #     nn.Linear(configs.seq_len, configs.pred_len),
        #     nn.ReLU(),
        #     # nn.Dropout(configs.dropout),
        #     nn.Linear(configs.pred_len, configs.pred_len),
        #     # nn.ReLU(),
        #     # nn.Linear(configs.d_model, configs.pred_len),
        # )
        

    
        self.conv_layers = []
        for i in range(len(conv_size)):
            self.conv_layers.append(ConvLayer(configs.c_out, conv_size[i]))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.conv_2d = nn.Sequential(
            Inception_Block_V1(len(conv_size)+1, configs.d_model, num_kernels=2),
            nn.Dropout(0.1),
            nn.GELU(),
            Inception_Block_V1(configs.d_model, 1, num_kernels=2),                                                                                                                                                                                                                                  
            # Inception_Block_V1(in_channels=len(conv_size)+1, out_channels=configs.d_model, num_kernels=6),
        )

        # self.conv_2d = nn.Sequential(
        #     # Inception_Block_V1(in_channels=len(conv_size)+1, out_channels=configs.d_model//2, num_kernels=2),
        #     # nn.Dropout(configs.dropout),
        #     # nn.GELU(),
        #     Inception_Block_V1(in_channels=len(conv_size)+1, out_channels=configs.d_model, num_kernels=2),
        #     nn.Dropout(configs.dropout),
        #     nn.GELU(),
        #     Inception_Block_V1(in_channels=configs.d_model, out_channels=configs.d_model//2, num_kernels=2),
        #     nn.Dropout(configs.dropout),
        #     nn.GELU(),
        #     Inception_Block_V1(in_channels=configs.d_model//2, out_channels=1, num_kernels=2),

        # )



        self.length_encoder = nn.Linear(configs.seq_len, configs.seq_len+configs.pred_len)
        # self.length_encoder = nn.Sequential(
        #     nn.Linear(configs.seq_len, configs.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(configs.dropout),
        #     nn.Linear(configs.d_model, configs.seq_len+configs.pred_len),
        # )


        self.length_decoder = nn.Linear(configs.seq_len+configs.pred_len, configs.pred_len)
        # self.length_decoder = nn.Sequential(
        #     nn.Linear(configs.seq_len+configs.pred_len, configs.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(configs.dropout),
        #     nn.Linear(configs.d_model, configs.pred_len),
        # )


        self.sensor_encoder = nn.Linear(configs.c_out, configs.d_model)
        self.sensor_decoder = nn.Linear(configs.d_model, configs.c_out)
        self.dropout = nn.Dropout(configs.dropout)





    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        
        
        means = batch_x.mean(1, keepdim=True).detach()
        batch_x = batch_x - means
        stdev = torch.sqrt(
            torch.var(batch_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        batch_x /= stdev


        B, T, N = batch_x.size()
        batch_x = batch_x.transpose(-1, -2)

        res = self.encoder(batch_x)
        batch_x = self.length_encoder(batch_x)
        

        


        temp_input = batch_x
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)

            # 线性插值
            # scale_out = F.interpolate(temp_input, size=self.configs.seq_len+self.configs.pred_len, mode="linear").unsqueeze(-1)

            # 三次样调插值
            scale_out = cubic_spline(temp_input.transpose(-1, -2), self.configs.seq_len+self.configs.pred_len).transpose(-1, -2).unsqueeze(-1)
            
            all_inputs.append(scale_out)


        # np.savetxt('./exp_ana_data/scale_ori.csv',batch_x.squeeze(0).squeeze(-1).transpose(0, 1).detach().cpu().numpy() ,delimiter=',')
        # np.savetxt('./exp_ana_data/scale_1.csv',all_inputs[0].squeeze(0).squeeze(-1).transpose(0, 1).detach().cpu().numpy() ,delimiter=',')
        # np.savetxt('./exp_ana_data/scale_2.csv',all_inputs[1].squeeze(0).squeeze(-1).transpose(0, 1).detach().cpu().numpy() ,delimiter=',')



        all_inputs = torch.cat(all_inputs, dim=-1)
        all_inputs = torch.cat([batch_x.unsqueeze(-1), all_inputs], dim=-1)
        all_inputs = all_inputs.permute(0, 3, 2, 1)

        conv_out = self.conv_2d(all_inputs).squeeze(1)

        conv_out = conv_out.permute(0, 2, 1)




        conv_out = batch_x + conv_out
        conv_out = self.dropout(conv_out)
        forecast = self.length_decoder(conv_out)

        
        forecast = forecast.transpose(-1, -2)
        forecast = forecast * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.configs.pred_len, 1))
        forecast = forecast + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.configs.pred_len, 1))


        return forecast


# ------------------------- 2D Convolutional Module ------------------------- #
class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


if __name__ == '__main__':
    # model = NBEATS(h=96, input_size=96, n_dim=8)
    #
    #
    # windows_batch = {}
    # windows_batch["insample_y"] = torch.rand(512, 8, 96)
    # # windows_batch["insample_mask"] = torch.ones(512, 8, 96)
    # output = model(windows_batch)
    # print(output.size())

    model = Model(h=192, input_size=96, n_dim=8)

    windows_batch = {}
    windows_batch["insample_y"] = torch.rand(512, 8, 96)
    windows_batch["insample_mask"] = torch.ones(512, 8, 96)
    output = model(windows_batch)

    print(output.size())
    
