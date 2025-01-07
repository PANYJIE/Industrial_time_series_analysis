import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import random
from .import Priorknowledge
import numpy as np

seed = 99
random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUSDE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(GRUSDE, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=0.1,
            num_layers=1,
            batch_first=True,
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(hidden_size, output_size)
        self.fc_sigma = nn.Linear(hidden_size, output_size)
        self.fc_noise_level = nn.Linear(hidden_size, 1)  # 额外的层用于动态噪声水平

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.gru.hidden_size).to(x.device)
        gru_out, _ = self.gru(x, h0)
        gru_out = self.bn(gru_out.transpose(1, 2)).transpose(1, 2)
        gru_out = self.dropout(gru_out)

        mu = self.fc_mu(gru_out)
        sigma = torch.nn.functional.softplus(self.fc_sigma(gru_out))

        # 动态计算噪声水平
        noise_level = torch.sigmoid(self.dropout(self.fc_noise_level(gru_out)))
        # noise_level = torch.sigmoid(self.fc_noise_level(gru_out))

        # 应用动态噪声水平
        adaptive_noise = noise_level * torch.randn_like(sigma)
        dt = 1.0  # 假设时间步长为1
        dW = adaptive_noise * torch.sqrt(torch.tensor(dt).to(x.device))
        sde_out = mu * dt + sigma * dW
        return sde_out


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, out_dim)
        self.linear_res = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layernorm = nn.LayerNorm(out_dim)

    def forward(self, x):
        # x: [B,L,in_dim] or [B,in_dim]
        # [B,L,in_dim] -> [B,L,hidden_dim] or [B,in_dim] -> [B,hidden_dim]
        h = F.relu(self.linear_1(x))
        # [B,L,hidden_dim] -> [B,L,out_dim] or [B,hidden_dim] -> [B,out_dim]
        h = self.dropout(self.linear_2(h))
        # [B,L,in_dim] -> [B,L,out_dim] or [B,in_dim] -> [B,out_dim]
        res = self.linear_res(x)
        out = self.layernorm(h + res)  # [B,L,out_dim] or [B,out_dim]
        # out: [B,L,out_dim] or [B,out_dim]
        return out


class AttentionFusion(nn.Module):
    def __init__(self, num_sensors):
        super(AttentionFusion, self).__init__()
        # num_sensors 是传感器的数量
        # 输入通道为2，因为我们将两个矩阵沿通道维拼接
        self.conv1 = nn.Conv2d(2, num_sensors, kernel_size=1)
        self.conv2 = nn.Conv2d(num_sensors, 1, kernel_size=1)

    def forward(self, A, B):
        # 增加一个维度来表示批量大小和通道，使得 A 和 B 变为 [1, 1, 14, 14]
        A = A.unsqueeze(0).unsqueeze(0)
        B = B.unsqueeze(0).unsqueeze(0)
        # 正确地沿通道维拼接 A 和 B
        x = torch.cat([A, B], dim=1)  # 现在 x 的维度为 [1, 2, 14, 14]
        x = F.relu(self.conv1(x))  # 应用第一个卷积层
        weights = torch.sigmoid(self.conv2(x))  # 应用第二个卷积层生成权重
        # 使用权重进行融合 A 和 B
        C = weights * A + (1 - weights) * B
        return C.squeeze(0).squeeze(0)  # 移除添加的额外维度


class Encoder(nn.Module):
    def __init__(self, p=0.3):
        super(Encoder, self).__init__()
        self.embedding = nn.Conv1d(in_channels=14, out_channels=14,
                                   kernel_size=1, stride=1, padding=0)
        self.GRUSDE = GRUSDE(14, 64, 14)
        self.channels = nn.Parameter(torch.rand(14, 14))
        with torch.no_grad():
            self.channels.fill_diagonal_(1)
        self.model = AttentionFusion(14).to(device)
        self.RB = ResidualBlock(420, 128, 64)
        self.dropout = torch.nn.Dropout(p=p, inplace=False)
        self.fc8 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.GRUSDE(x)
        # x, _ = self.gru(x)
        sensor_data = Priorknowledge.know_list
        adj = np.corrcoef(sensor_data)
        adj = torch.tensor(adj).to("cuda:0").to(torch.float32)
        C = self.model(self.channels.to(device), adj).to(device)
        x = torch.matmul(x, C)
        x = x.reshape(x.size(0), -1)
        x = self.RB(x)
        x = self.dropout(x)
        x = self.fc8(x)
        return x


if __name__ == '__main__':
    device = 'cuda:0'
    input = torch.ones([128, 30, 21]).to(device)
    encoder = Encoder().to(device)
    out = encoder(input)
