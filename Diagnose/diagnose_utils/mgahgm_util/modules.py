import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None , init_range=2.0, bias=True, device = "cpu"):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.init_range = init_range
        if output_size == None:
            output_size = input_size
        self.output_size = output_size

        self.encoder_h =nn.Linear(input_size, hidden_size, bias=bias)
        self.encoder_mu = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.encoder_log_variance = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.decoder_h = nn.Linear(hidden_size, output_size, bias=bias)
        self.decoder_mu = nn.Linear(output_size, output_size, bias=bias)
        self.decoder_log_variance = nn.Linear(output_size, output_size, bias=bias)

        self.device = device

        # 截断正态分布
        torch.nn.init.trunc_normal_(self.encoder_mu.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, a=-self.init_range, b=self.init_range)

    def forward(self, x, data):
        b = x.shape[0]
        n = x.shape[1]
        k = x.shape[2]
        # [256, 100, 102]
        encoder_h = F.tanh(self.encoder_h(x))
        encoder_mu = self.encoder_mu(encoder_h)
        encoder_log_variance = self.encoder_log_variance(encoder_h)
        encoder_variance = torch.exp(encoder_log_variance)

        encoder_scale = torch.sqrt(encoder_variance)
        epsilon_sampler = torch.torch.randn(b, n, self.hidden_size).to(self.device)
        z = encoder_mu + encoder_scale * epsilon_sampler

        # [256, 100, 51]
        decoder_h = F.tanh(self.decoder_h(z))
        decoder_mu = self.decoder_mu(decoder_h)
        decoder_log_variance = self.decoder_log_variance(decoder_h)
        decoder_variance = torch.exp(decoder_log_variance)
        # [b, n, z]
        feature_probability = self.pdf(x, decoder_mu, decoder_variance)
        # feature_probability = self.pdf(data, decoder_mu, decoder_variance)
        # [b, z]
        # reconstruction_log_probability = torch.sum(torch.log(feature_probability + 1e-35), 1)
        reconstruction_log_probability = torch.mean(torch.log(feature_probability + 1e-35), 1)
        # this is -Dkl formula
        minusDkl = torch.mean( ((1+torch.log(torch.square(encoder_scale))) - torch.square(encoder_mu)
                              - torch.square(encoder_scale))/2 )

        self.loss = -(torch.mean(reconstruction_log_probability) + minusDkl )

        # [b, z]
        return reconstruction_log_probability

    def pdf(self, data, mu, var):
        epsilon = 1e-14
        return torch.exp(-torch.pow(data-mu, 2) / (2.0 * (var+epsilon)) ) \
            / torch.sqrt(2.0 * math.pi * (var+epsilon) )


class AttentionMatrix(nn.Module):
    """多头注意⼒"""
    def __init__(self, input_size,   bias=False, **kwargs):
        super(AttentionMatrix, self).__init__(**kwargs)
        self.W_q = nn.Linear(input_size, input_size, bias=bias)
        self.W_k = nn.Linear(input_size, input_size, bias=bias)
    def forward(self, x):
        d = x.shape[-1]
        Q = self.W_q(x)
        K = self.W_k(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d)
        attention_weights = nn.functional.softmax(scores, dim=-1)
        return attention_weights

class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h, A):
        ## A: K X K  [k, k] 传感器数量
        ## H: N X K  X L X D [n, k, l, d] [256, 44, 60, 32]
        # h [b, n, k] , A [b, n, n]
        # h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        h_n = self.lin_n(torch.einsum('bnk,bnn->bnk',h,A))
        h_r = self.lin_r(h[:,:-1]) # [b, n-1, k]
        h_n[:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h.permute(0, 2, 1)

class GCN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GNN, self).__init__()
        self.lin = nn.Linear(input_size, hidden_size, bias=True)
        self.lin2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.lin3 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, h, A):
        # h [b, n, k]
        h = F.relu(self.lin(torch.einsum('bnk,bnn->bnk',h,A)))
        h = F.relu(self.lin2(torch.einsum('bnk,bnn->bnk',h,A)))
        h = self.lin3(h)
        return h


class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back



class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        #  args.feat_gat_embed_dim,
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        # [200, 1]
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)  # x shape (b, k, n)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # [256, 38, 100]                                      # (b, k, k, embed_dim)
            # a_input [256, 38, 38, 200]   vi 全连接 vj   相当于一个邻接矩阵图
            a_input = self._make_attention_input(Wx)  # (b, k, k, 2*embed_dim)
            # [256, 38, 38], a [200, 1] ,a是公式中的w
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        # [256, 38, 38]
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        # [256, 38, 100] (b, k, embed_dim), x [256, 38, 100]
        h = self.sigmoid(torch.matmul(attention, x))

        # [256, 100, 38]
        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        # [256, 1444, 100], v:[256, 38, 100]
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        # [256, 1444, 100]
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        # [256, 1444, 100]
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            # [256, 38, 38, 200]
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True, rel=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias
        self.rel = rel

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(window_size)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # Wx [256, 100, 38], x [256, 100, 38]           # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # [256, 100, 100, 76]    # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # [256,100,100] # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)  # [256,100,100]
        attention = torch.dropout(attention, self.dropout, train=self.training)
        if self.rel:
            h = self.relu(self.bn(torch.matmul(attention, x)))
        else:
            h = self.sigmoid(torch.matmul(attention, x))  # (b, n, k) [256, 100, 38]

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix [b, n*n, k]
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix [b, n*n, k]
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2) # [b, n*n, 2*k]

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class GRULayer(nn.Module):
    """Gated Recurrent Unit (GRU) Layer
    :param in_dim: number of input features
    :param hid_dim: hidden size of the GRU
    :param n_layers: number of layers in GRU
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(GRULayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.gru = nn.GRU(in_dim, hid_dim, num_layers=n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x, h0 =None):  # x [256, 100, 114]
        out, h = self.gru(x, h0)  # h [1, 256, 150]
        out, h = out, h[-1, :, :]  # h [256, 150] Extracting from last layer
        return out, h


class RNNDecoder(nn.Module):
    """GRU-based Decoder network that converts latent vector into output
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, n_layers, dropout):
        super(RNNDecoder, self).__init__()
        self.in_dim = in_dim
        self.dropout = 0.0 if n_layers == 1 else dropout
        self.rnn = nn.GRU(in_dim, hid_dim, n_layers, batch_first=True, dropout=self.dropout)

    def forward(self, x):
        # x [256, 100, 150]
        decoder_out, _ = self.rnn(x)
        return decoder_out


class ReconstructionModel(nn.Module):
    """Reconstruction Model
    :param window_size: length of the input sequence
    :param in_dim: number of input features
    :param n_layers: number of layers in RNN
    :param hid_dim: hidden size of the RNN
    :param in_dim: number of output features
    :param dropout: dropout rate
    """

    def __init__(self, window_size, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(ReconstructionModel, self).__init__()
        self.window_size = window_size
        self.decoder = RNNDecoder(in_dim, hid_dim, n_layers, dropout)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        # x will be last hidden state of the GRU layer
        h_end = x  # [256, 150]
        # [256, 100, 150]
        h_end_rep = h_end.repeat_interleave(self.window_size, dim=1).view(x.size(0), self.window_size, -1)
        # [256, 100, 150]
        decoder_out = self.decoder(h_end_rep)
        # [256, 100, 38]
        out = self.fc(decoder_out)
        return out


class Forecasting_Model(nn.Module):
    """Forecasting model (fully-connected network)
    :param in_dim: number of input features
    :param hid_dim: hidden size of the FC network
    :param out_dim: number of output features
    :param n_layers: number of FC layers
    :param dropout: dropout rate
    """

    def __init__(self, in_dim, hid_dim, out_dim, n_layers, dropout):
        super(Forecasting_Model, self).__init__()
        layers = [nn.Linear(in_dim, hid_dim)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))

        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.relu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


class GatTimeDecoder(nn.Module):
    """
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, out_dim, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(GatTimeDecoder, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias
        self.out_dim = out_dim

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # Wx [256, 100, 38], x [256, 100, 38]           # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)  # [256, 100, 100, 76]    # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # [256,100,100] # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)  # [256,100,100]
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))  # (b, n, k) [256, 100, 38]
        return self.fc(h)

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)

class GatFeatDecoder(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
        :param n_features: Number of input features/nodes
        :param window_size: length of the input sequence
        :param dropout: percentage of nodes to dropout
        :param alpha: negative slope used in the leaky rely activation function
        :param embed_dim: embedding dimension (output dimension of linear transformation)
        :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
        :param use_bias: whether to include a bias term in the attention layer
        """

    def __init__(self, n_features, window_size, out_dim, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(GatFeatDecoder, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        #  args.feat_gat_embed_dim,
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias
        self.out_dim = out_dim

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        # [200, 1]
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(n_features, out_dim)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)  # x shape (b, k, n)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)  # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))  # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)  # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)  # [256, 38, 100]                                      # (b, k, k, embed_dim)
            # a_input [256, 38, 38, 200]   vi 全连接 vj   相当于一个邻接矩阵图
            a_input = self._make_attention_input(Wx)  # (b, k, k, 2*embed_dim)
            # [256, 38, 38], a [200, 1] ,a是公式中的w
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)  # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        # [256, 38, 38]
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        # [256, 38, 100] (b, k, embed_dim), x [256, 38, 100]
        h = self.sigmoid(torch.matmul(attention, x))

        # [256, 100, 38]
        return self.fc(h.permute(0, 2, 1))
    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        # [256, 1444, 100], v:[256, 38, 100]
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        # [256, 1444, 100]
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        # [256, 1444, 100]
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            # [256, 38, 38, 200]
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)
