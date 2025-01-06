import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class PhyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, wind_fea_outsize):
        super(PhyGRU, self).__init__()
        self.coef_a = 10.104642
        self.coef_b = 2.137126

        self.layer_wind1 = nn.Linear(output_size, 128)
        self.layer_wind2 = nn.Linear(128, wind_fea_outsize)

        self.gru1 = nn.GRU(input_size, hidden_size, num_layers=2, dropout=0.2)

        self.layer_com1 = nn.Linear(hidden_size + wind_fea_outsize, 512)
        self.layer_com2 = nn.Linear(512, 256)
        self.layer_com3 = nn.Linear(256, output_size)

    # 注意，由于需要未来风速，所以需要用到target set中的数据
    def forward(self, x, target_set):
        # x [16, 1578, 3]   target [8, 1578, 4]
        # print(x.shape)
        # 找到target set中的原始风速数据
        wind_speed = target_set[:, :, 12]
        wind_speed = wind_speed.permute(1, 0)
        # 风速特征计算（检查计算是否正确）
        wind_fea = self.coef_a * (wind_speed ** self.coef_b)
        # 把机理计算的功率除以2000，防止数值过大
        wind_fea = wind_fea / 2000
        # 通过线性层
        w_fea = F.leaky_relu(self.layer_wind1(wind_fea))
        w_fea = F.leaky_relu(self.layer_wind2(w_fea)) #[1578, 16]
        output, h_n = self.gru1(x)
        h_n = h_n[-1] #[1578, 64]
        # 拼接lstm和经验公式的结果
        hidden = torch.cat([h_n, w_fea], 1)
        # [batch, hidden]-> [batch, output]
        hidden = F.leaky_relu(self.layer_com1(hidden))
        hidden = F.leaky_relu(self.layer_com2(hidden))
        out = F.leaky_relu(self.layer_com3(hidden))
        return out