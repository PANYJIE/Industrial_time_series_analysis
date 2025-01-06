import torch.nn as nn
import torch
from . import NF, graph_layer, modules
MAF = NF.MAF
GraphLayer = graph_layer.GraphLayer
GNNLayer = graph_layer.GNNLayer
ConvLayer = modules.ConvLayer
GRULayer = modules.GRULayer
TemporalAttentionLayer = modules.TemporalAttentionLayer
Forecasting_Model = modules.Forecasting_Model
ReconstructionModel = modules.ReconstructionModel
GatTimeDecoder = modules.GatTimeDecoder
GatFeatDecoder = modules.GatFeatDecoder
AttentionMatrix = modules.AttentionMatrix
GNN = modules.GNN
VAE = modules.VAE


# from MTGFlow_NF import MAF
# from ts_flows import MAF


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class GatGame(nn.Module):
    def __init__(
            self,
            n_features,
            window_size,
            out_dim,
            kernel_size=7,
            time_gat_embed_dim=None,
            use_gatv2=True,
            dropout=0.2,
            alpha=0.2,
            mask_rate=0.3,
            replace_rate=0,
            device="cpu",
            convEn=True,
            remaskEn=True,
            batch_size=None,
            topk=15,
            flow_hidden_size=32,
            flow_n_hidden=1,
            flow_n_blocks=1,
            rel=True
    ):
        super(GatGame, self).__init__()

        # self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = GNNLayer(window_size, window_size, heads=1, rel=rel)
        self.temporal_gat = TemporalAttentionLayer(
            n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2, rel=rel)

        ''' flow '''
        # self.flow = MAF(n_blocks=flow_n_blocks, input_size=window_size, hidden_size=flow_hidden_size,
        #             n_hidden=flow_n_hidden, cond_label_size=window_size*2, activation='tanh')
        self.flow = MAF(n_blocks=int(flow_n_blocks), input_size=1, hidden_size=flow_hidden_size,
                        n_hidden=int(flow_n_hidden), cond_label_size=1 * 2, activation='tanh')
        # self.flow = MAF(n_blocks=1, n_sensor=n_features, input_size=1, hidden_size=32,
        #                 n_hidden=1, cond_label_size=2, batch_norm=True,activation='tanh')

        ''' vae '''
        self.vae = VAE(input_size=n_features, hidden_size=2 *
                       n_features, output_size=n_features, device=device)
        self.vae_input = nn.Linear(2*n_features, n_features)

        self.mask_rate = mask_rate
        self.device = device
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        # mask token
        self.enc_feat_mask_token = nn.Parameter(torch.zeros(1, window_size))
        self.enc_time_mask_token = nn.Parameter(torch.zeros(1, n_features))

        self.convEn = convEn
        self.remaskEn = remaskEn

        self.embedding = nn.Embedding(n_features, window_size)
        self.n_features = n_features
        self.window_size = window_size
        self.topk = topk
        self.batch_size = batch_size

    def forward(self, x, train=True):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        b = x.shape[0]
        n = x.shape[1]
        k = x.shape[-1]
        # if self.convEn:
        #     x = self.conv(x)
        ''' feature '''
        x_feat, mask_nodes_feat, keep_nodes_feat = None, None, None
        if train:
            x_feat, mask_nodes_feat, keep_nodes_feat = self.encoding_mask_feature(
                x)
        else:
            x_feat = x

        # 求 topk 数组 # [51, 100]
        all_embeddings = self.embedding(
            torch.arange(self.n_features).to(self.device))
        # weights_arr 用来构边，先detach()
        weights_arr = all_embeddings.detach().clone()  # [51, 100]
        all_embeddings = all_embeddings.repeat(b, 1)  # [13056, 100]
        weights = weights_arr.view(self.n_features, -1)  # [51, 100]
        cos_ji_mat = torch.matmul(weights, weights.T)  # [51, 51]
        normed_mat = torch.matmul(weights.norm(
            dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))  # [51, 51]
        cos_ji_mat = cos_ji_mat / normed_mat  # [51, 51]
        topk_num = self.topk

        # 通常该函数返回2个值，第一个值为排序的数组，第二个值为该数组中获取到的元素在原数组中的位置标号。
        topk_indices_ji = torch.topk(
            cos_ji_mat, topk_num, dim=-1)[1]  # [51, topk_num]
        self.learned_graph = topk_indices_ji
        # [1, n_features*topk_num]
        gated_i = torch.arange(0, self.n_features).T.unsqueeze(
            1).repeat(1, topk_num).flatten().to(self.device).unsqueeze(0)
        # [1, n_features*topk_num]
        gated_j = topk_indices_ji.flatten().unsqueeze(0)
        # [2, n_features*topk_num]
        gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
        # [2, n_features*topk_num*b]
        batch_gated_edge_index = get_batch_edge_index(
            gated_edge_index, b, k).to(self.device)

        # [b*k, n]
        x_feat = x_feat.permute(
            0, 2, 1).contiguous().view(-1, self.window_size)
        # [b*k, n]
        h_feat = self.feature_gat(
            x_feat, batch_gated_edge_index, embedding=all_embeddings)
        # [b, k, n]
        h_feat = h_feat.view(-1, self.n_features, self.window_size)
        # [b, n, k]
        h_feat = h_feat.permute(0, 2, 1).contiguous()
        # remask
        if self.remaskEn and train:
            h_feat = h_feat.clone().permute(0, 2, 1).reshape(b * k, n)
            h_feat[mask_nodes_feat] = 0.0
            h_feat = h_feat.reshape(b, k, n).permute(0, 2, 1)

        ''' time '''
        x_time, mask_nodes_time, keep_nodes_time = None, None, None
        if train:
            x_time, mask_nodes_feat, keep_nodes_feat = self.encoding_mask_feature(
                x)
        else:
            x_time = x

        h_time = self.temporal_gat(x_time)  # (b, n, k)
        # remask
        if self.remaskEn and train:
            h_time = h_time.clone().reshape(b * n, k)
            h_time[mask_nodes_time] = 0.0
            h_time = h_time.reshape(b, n, k)

        ''' flow '''
        # hidden = torch.cat((h_feat, h_time), 1)
        # flow_log_prob = self.flow.log_prob(x.permute(0, 2, 1).reshape(-1, n), hidden.permute(0, 2, 1).reshape(-1, 2*n)).reshape(-1, k)
        hidden = torch.cat((h_feat, h_time), -1)
        flow_log_prob = self.flow.log_prob(
            x.reshape(-1, 1), hidden.reshape(-1, 2)).reshape((-1, n, k))
        # [b, z]
        # log_prob = self.flow.log_prob(x.reshape(-1, 1), k, n , hidden.reshape(-1, 2))
        # log_prob = self.flow.log_prob(x.reshape(-1, 1), k, n)
        self.flow_loss = -flow_log_prob.mean()
        if flow_log_prob.dim() == 3:
            flow_log_prob = torch.mean(flow_log_prob, 1)

        '''vae'''
        # hidden = torch.cat((h_feat, h_time), -1)
        vae_input = self.vae_input(hidden)
        vae_log_prob = self.vae(vae_input, x)
        self.vae_loss = self.vae.loss
        self.total_loss = self.flow_loss + self.vae_loss

        # [b, k]
        return flow_log_prob, vae_log_prob

    def encoding_mask_feature(self, x):
        '''

        :param x:
        :return:
        x_feat：掩码后的x
        mask_nodes：掩码节点的索引
        keep_nodes：未掩码节点的索引
        '''
        b = x.shape[0]
        n = x.shape[1]
        k = x.shape[-1]
        x_feat = x.permute(0, 2, 1).clone()  # (b, k, n)
        x_feat = x_feat.reshape(b * k, n)  # (b*k, n)

        num_nodes = b * k
        # random masking
        num_mask_nodes = int(self.mask_rate * num_nodes)
        # perm = torch.randperm(num_nodes, device=self.device)
        perm = torch.randperm(num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            # perm_mask = torch.randperm(num_mask_nodes, device=self.device)
            perm_mask = torch.randperm(num_mask_nodes)
            token_nodes = mask_nodes[perm_mask[: int(
                self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(
                self._replace_rate * num_mask_nodes):]]
            # noise_to_be_chosen = torch.randperm(num_nodes, device=self.device)[:num_noise_nodes]
            noise_to_be_chosen = torch.randperm(num_nodes)[:num_noise_nodes]
            x_feat[token_nodes] = 0.0
            x_feat[noise_nodes] = x_feat[noise_to_be_chosen]
        else:
            token_nodes = mask_nodes
            x_feat[mask_nodes] = 0.0

        x_feat[token_nodes] = x_feat[token_nodes] + self.enc_feat_mask_token

        x_feat = x_feat.reshape(b, k, n)
        x_feat = x_feat.permute(0, 2, 1)
        return x_feat, mask_nodes, keep_nodes

    def encoding_mask_time(self, x):
        b = x.shape[0]
        n = x.shape[1]
        k = x.shape[-1]

        x_time = x.reshape(b * n, k)  # (b*n, k)

        num_nodes = b * n
        # random masking
        num_mask_nodes = int(self.mask_rate * num_nodes)
        perm = torch.randperm(num_nodes, device=self.device)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=self.device)
            token_nodes = mask_nodes[perm_mask[: int(
                self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(
                self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=self.device)[
                :num_noise_nodes]
            x_time[token_nodes] = 0.0
            x_time[noise_nodes] = x_time[noise_to_be_chosen]
        else:
            token_nodes = mask_nodes
            x_time[mask_nodes] = 0.0

        x_time[token_nodes] = x_time[token_nodes] + self.enc_time_mask_token

        x_time = x_time.reshape(b, n, k)
        return x_time, mask_nodes, keep_nodes
