import torch
import torch.nn as nn
from math import pi
from torchdiffeq import odeint, odeint_adjoint
#from vector_fields import SingleHiddenLayer, FinalTanh, GRU_ODE,FinalTanh_f,FinalTanh_ff,FinalTanh_ff6,FinalTanh_f,FinalTanh_g,FinalTanh_hide
from einops import rearrange, repeat, reduce
# from ode_func import ODEFunc, ODEFunc_w_Poisson, ODEFunc_Attention

# import utils as utils
# device = 'cuda:1' if(torch.cuda.is_available()) else 'cpu'

MAX_NUM_STEPS = 1000  # Maximum number of steps for ODE solver

#### 一般func
class ODEFunc(nn.Module):
    """MLP modeling the derivative of ODE system.

    Parameters
    ----------
    device : torch.device

    data_dim : int
        Dimension of data.

    hidden_dim : int
        Dimension of hidden layers.

    augment_dim: int
        Dimension of augmentation. If 0 does not augment ODE, otherwise augments
        it with augment_dim dimensions.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, device, data_dim, hidden_dim, augment_dim=0,
                 time_dependent=False, non_linearity='relu'):
        super(ODEFunc, self).__init__()
        self.device = device
        self.augment_dim = augment_dim
        self.data_dim = data_dim
        self.input_dim = data_dim + augment_dim
        self.hidden_dim = hidden_dim
        self.nfe = 0  # Number of function evaluations
        self.time_dependent = time_dependent

        if time_dependent:
            self.fc1 = nn.Linear(self.input_dim + 1, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.input_dim)

        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU(inplace=True)
        elif non_linearity == 'softplus':
            self.non_linearity = nn.Softplus()

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time. Shape (1,).

        x : torch.Tensor
            Shape (batch_size, input_dim) ([128, 100])
        """
        # Forward pass of model corresponds to one function evaluation, so
        # increment counter
        # print("anode/ODEFunc x shape is {}".format(x.shape)) # ([128, 100]
        self.nfe += 1
        if self.time_dependent:
            # Shape (batch_size, 1)
            t_vec = torch.ones(x.shape[0], 1).to(self.device) * t
            # Shape (batch_size, data_dim + 1)
            t_and_x = torch.cat([t_vec, x], 1)
            # Shape (batch_size, hidden_dim)
            out = self.fc1(t_and_x)
        else:
            out = self.fc1(x)
        out = self.non_linearity(out)  # [128, 100]
        # print("82 anode/ODEFunc out shape is{} ".format(out.shape))
        out = self.fc2(out)
        # print("84 anode/ODEFunc out shape is{} ".format(out.shape))
        out = self.non_linearity(out)
        # print("86 anode/ODEFunc out shape is{} ".format(out.shape))
        out = self.fc3(out)
        # print("88 anode/ODEFunc out shape is{} ".format(out.shape))
        return out


class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.

    Parameters
    ----------
    device : torch.device

    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.

    is_conv : bool
        If True, treats odefunc as a convolutional model.

    tol : float
        Error tolerance.

    adjoint : bool
        If True calculates gradient with adjoint method, otherwise
        backpropagates directly through operations of ODE solver.
    """
    def __init__(self, device, odefunc, is_conv=False, tol=1e-3, adjoint=False):
        super(ODEBlock, self).__init__()
        self.adjoint = adjoint
        self.device = device
        self.is_conv = is_conv
        self.odefunc = odefunc
        self.tol = tol

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)  ([128, 100])

        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """
        # Forward pass corresponds to solving ODE, so reset number of function
        # evaluations counter
        self.odefunc.nfe = 0

        # print("ODEBlock x shape is {}".format(x.shape))

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)


        if self.odefunc.augment_dim > 0:
            if self.is_conv:
                # Add augmentation
                batch_size, channels, height, width = x.shape
                aug = torch.zeros(batch_size, self.odefunc.augment_dim,
                                  height, width).to(self.device)
                # Shape (batch_size, channels + augment_dim, height, width)
                x_aug = torch.cat([x, aug], 1)
            else:
                # Add augmentation
                aug = torch.zeros(x.shape[0], self.odefunc.augment_dim).to(self.device)
                # Shape (batch_size, data_dim + augment_dim)
                x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        if self.adjoint:
            out = odeint_adjoint(self.odefunc, x_aug, integration_time,
                                 rtol=self.tol, atol=self.tol, method='dopri5',
                                 options={'max_num_steps': MAX_NUM_STEPS})
        else:
            # print("x_aug shape is {}".format(x_aug.shape))  #[16, 100, 64])
            # print("inter_time shape is {}".format(integration_time.shape)) #[2]
            out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method='dopri5',
                         options={'max_num_steps': MAX_NUM_STEPS})

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)






# class ODENet(nn.Module):
#     """An ODEBlock followed by a Linear layer.

#     Parameters
#     ----------
#     device : torch.device

#     data_dim : int
#         Dimension of data.

#     hidden_dim : int
#         Dimension of hidden layers.

#     output_dim : int
#         Dimension of output after hidden layer. Should be 1 for regression or
#         num_classes for classification.

#     augment_dim: int
#         Dimension of augmentation. If 0 does not augment ODE, otherwise augments
#         it with augment_dim dimensions.

#     time_dependent : bool
#         If True adds time as input, making ODE time dependent.

#     non_linearity : string
#         One of 'relu' and 'softplus'

#     tol : float
#         Error tolerance.

#     adjoint : bool
#         If True calculates gradient with adjoint method, otherwise
#         backpropagates directly through operations of ODE solver.
#     """
#     def __init__(self, device, data_dim, hidden_dim, output_dim=100,
#                  augment_dim=0, time_dependent=False, non_linearity='relu',
#                  tol=1e-3, adjoint=False):
#         super(ODENet, self).__init__()
#         self.device = device
#         self.data_dim = data_dim
#         self.hidden_dim = hidden_dim
#         self.augment_dim = augment_dim
#         self.output_dim = output_dim
#         self.time_dependent = time_dependent
#         self.tol = tol
#         self.LinearAttention = LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
#                                       dim_head=hidden_dim, dropout=0.1,
#                                       relative_emb=False,
#                                       #relative_emb=True,
#                                       init_method='xavier',
#                                       init_gain=1.
#             )
#         self.project = nn.Linear(hidden_dim, 100)

#         odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim,
#                           time_dependent, non_linearity)

#         self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
#         # self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim,
#         #                               self.output_dim)

#     def forward(self,k, v,  return_features=False):     ### 改了
#         q_h = k.clone()
#         k_h = k.clone()
#         v_h = v.clone()

#         # print("anode/odenet x shape is {}".format(x.shape))
#         update, _ = self.LinearAttention(q_h, k_h, v_h)
#         up_h = self.project(update)
#         features = self.odeblock(up_h)  # [128, 100]

#         # print("238anode/ODENet/features:{}".format(features.shape))
#         # pred = self.linear_layer(features)
#         # if return_features:
#         #     return features, pred
#         return features


class ODENet2D(nn.Module):
    def __init__(self, device, data_dim, hidden_dim, output_dim=100,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(ODENet2D, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol
        self.vh_unembedding_layer = nn.Linear(100, 6000, bias=False)
        self.LinearAttention = LinearAttention(input_dim=hidden_dim, attn_type='galerkin',
                                      dim_head=hidden_dim, dropout=0.1,
                                      relative_emb=False,
                                      #relative_emb=True,
                                      init_method='xavier',
                                      init_gain=1.
            )
        self.project = nn.Linear(hidden_dim, 100)

        odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim,
                          time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
        # self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim,
        #                               self.output_dim)

    def forward(self, k, v, return_features=False):
        q_h = k.clone()
        k_h = k.clone()
        v_h = v.clone()    # [16,100,64]

        # print("anode/odenet x shape is {}".format(x.shape))

        update, _ = self.LinearAttention(q_h, k_h, v_h)   # [16,100,64]

        ### 2D
        # print("anode/odenet update shape is {}".format(update.shape))  # ([16, 100, 100])
        # electicity 数据集update需要reshape
        # up_h = update.reshape(16, -1)  # [16, 10000]   
        # print("up_h shape is {}".format(up_h.shape))
        
        # NS不需要update进行reshape

        features = self.odeblock(update)  # [128, 100]
        # print("features shape is {}".format(features.shape)) # [16, 100, 100]
        # print("238anode/ODENet/features:{}".format(features.shape))
        # pred = self.linear_layer(features)
        # if return_features:
        #     return features, pred
        return features


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

        #print("\nINPUT DIM\n")
        self.to_q = nn.Linear(input_dim, inner_dim, bias = False)
        self.to_k = nn.Linear(input_dim, inner_dim, bias = False)
        self.to_v = nn.Linear(input_dim, inner_dim, bias = False)

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
                #nn.Linear(input_dim, input_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                #nn.Linear(inner_dim + pos_dim*heads, input_dim),
                nn.Linear(inner_dim, input_dim),
                #nn.Linear(input_dim, input_dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / np.sqrt(dim_head)
            self.diagonal_weight = 1. / np.sqrt(dim_head)
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        #if init_params:
        #    self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        #for param in self.to_qkv.parameters():
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)

                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1),
                                                                                    dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)


    def forward(self, queries, keys, values, pos=None, not_assoc=False, padding_mask=None):
        if pos is None and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')

        queries = self.to_q(queries)
        keys = self.to_k(keys)
        values = self.to_k(values)
        queries, keys, values = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (queries,keys,values))
        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(keys, self.k_norm)
                v = self.norm_wrt_domain(values, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(queries, self.q_norm)
                k = self.norm_wrt_domain(keys, self.k_norm)
        else:
            grid_size = torch.sum(padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)  # [b, 1, 1]

            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]

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

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]


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
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')

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
                padding_mask_arr = torch.matmul(padding_mask, padding_mask.transpose(-1, -2))  # [b, h, n, n]
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
                #print(k.shape)
                #print(v.shape)
                #print(queries.shape)
                dots = torch.matmul(keys.transpose(-1, -2), values)
                out = torch.matmul(queries, dots) * (1./queries.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        #print(out.shape)
        #print(self.to_out)
        #return self.to_out(out), None
        #return self.to_out(out), dots
        #return self.to_out(out), dots
        #print(self.to_k.weight)
        return self.to_out(out), self.to_q.weight


#### no attention
class ODENet(nn.Module):
 
    def __init__(self, device, data_dim, hidden_dim, output_dim=1,
                 augment_dim=0, time_dependent=False, non_linearity='relu',
                 tol=1e-3, adjoint=False):
        super(ODENet, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.augment_dim = augment_dim
        self.output_dim = output_dim
        self.time_dependent = time_dependent
        self.tol = tol

        odefunc = ODEFunc(device, data_dim, hidden_dim, augment_dim,
                          time_dependent, non_linearity)

        self.odeblock = ODEBlock(device, odefunc, tol=tol, adjoint=adjoint)
        # self.linear_layer = nn.Linear(self.odeblock.odefunc.input_dim,
                                    #   self.output_dim)


    def forward(self, x, return_features=False):
        features = self.odeblock(x)
        # print("x shape is ",x.shape)
        # print(features.shape)
        # pred = self.linear_layer(features)
        # print(pred.shape)
        # if return_features:
            # return features, pred
        return features