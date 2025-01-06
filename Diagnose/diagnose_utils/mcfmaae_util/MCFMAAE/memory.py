import torch.nn as nn
import torch
import torch.nn.functional as F


class DenseEncoder(nn.Module):
    def __init__(self, data_size, hidden_size, latent_size):
        super(DenseEncoder, self).__init__()
        self.__dict__.update(locals())

        self.layers = nn.Sequential(
            nn.Linear(data_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, latent_size)
        )

    def forward(self, x):
        out = self.layers(x)

        return out


class MemoryModule(nn.Module):
    def __init__(self, mem_size, hidden_size, latent_size, batch_size=128):
        super(MemoryModule, self).__init__()

        self.memory_bank = nn.Parameter(torch.randn(mem_size, latent_size))
        self.pred_net = DenseEncoder(latent_size, hidden_size, mem_size)

    def forward(self, z):
        weights = self.pred_net(z)
        z = F.softmax(weights, dim=-1)
        z_mem = torch.einsum('ij,bki->bkj', self.memory_bank, z)

        return z_mem
