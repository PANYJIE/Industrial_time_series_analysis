#按phy设计的两个gcn加在一起放进seq的de上  验证v在前面是否要经过gcn，验证两个gcn是不是不同层效果更好,目前来说最好
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from Industrial_time_series_analysis.Forecast.forecast_utils.STD_Phy_util.net import gated_selfatt_T



#第一个seq2seq 16，4，13变成16，4，12，不需要seq2seq，直接一个lstm。第二个需要seq2seq，16，12，4 变成16，12，4
class Encoder(nn.Module):
    def __init__(self, input_size,  hidden_size, output_size=12,device='cuda:0'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_directions = 1
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h,c):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(self.device)
        # input_seq(batch_size, input_size)
        # output, (h, c) = self.lstm(input_seq)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, 1, hidden_size)
        pred = self.linear(output.squeeze(1))  # pred(batch_size, 1, output_size)变成pred(batch_size, output_size)

        return pred,h, c

class Decoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=16,mlp_output_size=4, output_size=4,device='cuda:0'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.output_size = output_size
        self.num_directions = 1
        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output.squeeze(1))  # pred(batch_size, 1, output_size)变成pred(batch_size, output_size)

        return pred, h, c
    

class gated_final(nn.Module):

    def __init__(self, input_size=4, hidden_size=16,mlp_output_size=4, output_size=4,seq_out_len=12,device='cuda:0'):
        super(gated_final, self).__init__()
        self.seq_out_len = seq_out_len
        self.W_f = nn.Linear(seq_out_len, seq_out_len)
        self.W_t = nn.Linear(seq_out_len, seq_out_len)

    def forward(self, flow_input, tuan_input):
        #16,1,4,12变成64，12
        b,w,node,out_len = flow_input.size()[0],flow_input.size()[1],flow_input.size()[2],flow_input.size()[3]
        flow_input = flow_input.reshape(-1,self.seq_out_len)
        tuan_input = tuan_input.reshape(-1,self.seq_out_len)
        z = torch.sigmoid(self.W_f(flow_input)+self.W_t(tuan_input))
        hidden_state = torch.mul(z, flow_input)+torch.mul(1-z, tuan_input)
        output = hidden_state
        output = output.reshape(b,w,node,out_len) 
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size=4, seq_len=12, hidden_size=16,mlp_output_size=4,seq_out_len=12, output_size=4,device='cuda:0',a_dir = "G",dropout=0.5):
        super(LSTMModel, self).__init__()

        self.seq_out_len = seq_out_len
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.output_size = output_size
        self.seq_len = seq_len
        self.a_dir = a_dir
        self.dropout = dropout

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)#输入（b,11,4）out=b,11,hidden_size
        #得到每个传感器的风俗初步表达
        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.ReLU(),
            nn.Linear(8, mlp_output_size)
        )#弄到这里 b,4
      

        # Seq2Seq layer (if needed)  这里需要把风速加入浓度，直接拼接，然后序列这里学习的是风速方向的序列（b,4,13）
        #self.seq2seq = nn.LSTM(output_size, hidden_size, batch_first=True)#b,4,12+outsize
        #self.seq2seq2 = nn.LSTM(output_size, hidden_size, batch_first=True)
        #这里需要改一下
        if self.a_dir == "L":
            self.seq2seq = nn.LSTM(seq_len+1, hidden_size, batch_first=True)
            self.linear_layer=nn.Linear(hidden_size,seq_len)#v_out=b,4,12
        if self.a_dir == "G":
            self.GCN = gcn(1, 1, dropout, support_len=1, order = 2)
            self.linear_layer = nn.Linear(seq_len+1,seq_len)
        #改到这里
            self.GCN2 = gcn(1, 1, dropout, support_len=1, order = 2)
            # self.GCN3 = gcn(1, 1, dropout, support_len=1, order = 2)
            # self.linear_layer2 = nn.Linear(seq_len+1,seq_len)
            # self.bias = nn.Parameter(torch.randn(input_size, seq_len), requires_grad=True)



        #b,4,12-b,12,4

        self.encoder2 = Encoder(input_size, hidden_size)

        # Decoder2 
        self.decoder2 = Decoder(input_size, hidden_size)
       
        #b,12,4



    def forward(self, x , z):
        # LSTM layer
        #z=b,11,4  x=b,12,4

        lstm_out, _ = self.lstm(z)
        b, s, h = lstm_out.shape#b,11,4

        # MLP layer
        
        mlp_out = self.mlp(lstm_out[:, -1, :])#b,4

        x_reshaped = x.reshape(x.shape[0], x.shape[2], x.shape[1])#b,4,12

        v_qian = 1#将速度先通过一个gcn
        if v_qian == 1:
            adjacency_matrix = torch.tensor([
                                [0, 1, 0, 0],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0]
                            ], dtype=torch.float32)
            support = [adjacency_matrix.to(self.device)]
            mlp_out = mlp_out.unsqueeze(1).unsqueeze(3)#b,1,4,1
            mlp_out = self.GCN2(mlp_out,support)
            mlp_out = mlp_out.squeeze()#b,4

        seq_input1 = torch.cat([x_reshaped, mlp_out.unsqueeze(2)], dim=2)#b,4,13

        if self.a_dir == "L":

            v_out, _ = self.seq2seq(seq_input1)#v_out=b,4,16
            s, b, h = v_out.shape
            v_out = v_out.reshape(s * b, h)  # 转换成线性层的输入格式
            v_out=self.linear_layer(v_out)
            v_out= v_out.reshape(s, b, -1)#v_out=b,4,12
        #风速加入与传感器序列挖掘结束
        if self.a_dir == "G":
            adjacency_matrix = torch.tensor([
                                [0, 1, 0, 0],
                                [1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0]
                            ], dtype=torch.float32)
            support = [adjacency_matrix.to(self.device)]
            # support = [torch.eye(self.num_nodes, self.num_nodes).to(self.device)+F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)]
            v_out = seq_input1.unsqueeze(1)#b,1,4,13
            s, v, b, h = v_out.shape
            v_out = v_out.reshape(s * b * v, h)
            v_out=self.linear_layer(v_out)
            v_out= v_out.reshape(s, v, b, -1)
            # v_out=self.GCN(v_out,support)
            v_out = v_out.squeeze(1)#b,4,12  将风速与特征进行融合


            

        input_seq_v = v_out.permute(0, 2, 1)#b,12,4
        input_seq = x_reshaped.permute(0, 2, 1)#b,12,4
        #这里开始改
        target_len = self.seq_out_len  # 预测步长
        batch_size, seqlen, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        # encoder_input = input_seq[:, 0, :]
        outputs = torch.zeros(batch_size, self.input_size, self.seq_len).to(self.device)
        h = torch.zeros(1, batch_size, self.hidden_size).to(self.device)#torch.randn(self.num_directions * self.num_layers 就是1, batch_size, self.hidden_size).to(self.device)
        c = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        # h_v = torch.zeros(1, batch_size, self.hidden_size).to(self.device)#torch.randn(self.num_directions * self.num_layers 就是1, batch_size, self.hidden_size).to(self.device)
        # c_v = torch.zeros(1, batch_size, self.hidden_size).to(self.device)
        for t in range(seqlen):
            
            # h = h.squeeze().unsqueeze(1).unsqueeze(3)#b,1,4,1
            # h = self.GCN(h,support)#b,1,4,1
            encoder_input = input_seq[:, t, :]
            encoder_input = encoder_input.unsqueeze(1).unsqueeze(3)#b,1,4,1
            encoder_input = self.GCN(encoder_input,support)#b,1,4,1

            encoder_input_v = input_seq_v[:, t, :]
            encoder_input_v = encoder_input_v.unsqueeze(1).unsqueeze(3)#b,1,4,1
            encoder_input_v = self.GCN2(encoder_input_v,support)#b,1,4,1
            # h = h.squeeze(3).permute(1,0,2)#1,b,4
            encoder_input = encoder_input.squeeze()
            encoder_input_v = encoder_input_v.squeeze()
            en_input = encoder_input + encoder_input_v
            encoder_output, h, c = self.encoder2(en_input, h, c)
            outputs[:, :, t] = encoder_output#b,outputsize,seq_out_len


        outputs2 = torch.zeros(batch_size, self.input_size, self.seq_len).to(self.device)
        decoder_input = input_seq[:, -1, :]#16,4
        decoder_input_v = input_seq_v[:, -1, :]#16,4
        de_input = decoder_input + decoder_input_v

        for t in range(target_len):
            # h = h.squeeze().unsqueeze(1).unsqueeze(3)#b,1,4,1
            # h = self.GCN2(h,support)#b,1,4,1

            # decoder_input = decoder_input.unsqueeze(1).unsqueeze(3)#b,1,4,1
            # decoder_input = self.GCN3(decoder_input,support)#b,1,4,1
            # h = h.squeeze(3).permute(1,0,2)#1,b,4
            # decoder_input = decoder_input.squeeze()
            decoder_output, h, c = self.decoder2(de_input, h, c)
            outputs2[:, :, t] = encoder_output#b,outputsize,seq_out_len
            de_input = decoder_output

        b, out_size, seq_outlen = outputs2.shape[0], outputs2.shape[1], outputs2.shape[2]
        outputs2 = outputs2.reshape(b, seq_outlen, -1)#b,12,4 

        


        return outputs2#b,seq_out_len,outputsize




class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))#(16,32,170,29)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):#？
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]#(16,32,170,29)
        for a in support:
            x1 = self.nconv(x, a)#(16,32,170,29)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h#(16,32,170,29)


class final_ae_stcnn(nn.Module):#编码器与解码器结构
    def __init__(self, num_nodes, seq_len, node_dim, blocks=4, layers=2, kernel_size=2, dropout=0.5,
                 channels=32, out_dim=2, device='cuda:0'):
        super(final_ae_stcnn, self).__init__()
        self.num_nodes = num_nodes
        self.blocks = blocks
        self.layers = layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.device = device

        self.filter_convs = nn.ModuleList()#创建一个空的nn.ModuleList对象，用于存放卷积层
        self.filter_convs_trans = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.gate_convs_trans = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.residual_convs_trans = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.skip_convs_trans = nn.ModuleList()
        self.res_map = nn.ModuleList()
        self.res_map_trans = nn.ModuleList()
        self.skip_map = nn.ModuleList()
        self.skip_map_trans = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.bn_trans = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.gconv_trans = nn.ModuleList()

        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, node_dim).to(device), requires_grad=True).to(device)#是一个可训练参数，需要计算梯度
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, num_nodes).to(device), requires_grad=True).to(device)
        self.nodevec1_trans = nn.Parameter(torch.randn(num_nodes, node_dim).to(device), requires_grad=True).to(device)
        self.nodevec2_trans = nn.Parameter(torch.randn(node_dim, num_nodes).to(device), requires_grad=True).to(device)

        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))#channels：输出通道数。这里指的是输出张量的通道数，即卷积核的数量
        self.start_conv_trans = nn.ConvTranspose2d(channels, 1, kernel_size=(1, 1))

        receptive_field = 1#初始化感受野大小为1。？

        for b in range(blocks):#循环blocks次，每次执行一组卷积操作。
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):#循环layers次，每次执行一次卷积操作
                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size),dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.filter_convs_trans.append(nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs_trans.append(nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))
                self.residual_convs_trans.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))  # focus on this "trans"
                self.skip_convs.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))
                self.skip_convs_trans.append(nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=(1, 1)))
                self.res_map.append(nn.Linear(in_features=receptive_field+kernel_size, out_features=receptive_field+new_dilation-1))#？
                self.res_map_trans.append(nn.Linear(in_features=receptive_field+new_dilation-1, out_features=receptive_field+kernel_size))
                self.skip_map.append(nn.Linear(in_features=receptive_field+kernel_size, out_features=receptive_field+new_dilation-1))
                self.skip_map_trans.append(nn.Linear(in_features=receptive_field+new_dilation-1, out_features=receptive_field+kernel_size))
                self.bn.append(nn.BatchNorm2d(channels))
                self.bn_trans.append(nn.BatchNorm2d(channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.gconv.append(gcn(channels, channels, dropout, support_len=1))
                self.gconv_trans.append(gcn(channels, channels, dropout, support_len=1))

        self.end_conv = nn.Conv2d(in_channels=channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)#最后再卷积一下
        self.end_conv_trans = nn.ConvTranspose2d(in_channels=out_dim, out_channels=channels, kernel_size=(1, 1), bias=True)

        self.receptive_field = receptive_field


    def encoder(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)16,1,170,30
        :return: (batch_size, 1, node_num, predict_length)
        """
        inputs = inputs[:, :1, :, :]

        support = [torch.eye(self.num_nodes, self.num_nodes).to(self.device)+F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)]
        #170*170
        in_len = inputs.size(3)#30
        if in_len < self.receptive_field:
            x = nn.functional.pad(inputs, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = inputs#(16,1,170,30)
        x = self.start_conv(x)#(16,32,170,30)
        skip = 0
        for i in range(self.blocks * self.layers):
            residual = x#(16,32,170,30)
            # dilated convolution
            aa = self.blocks * self.layers-i-1#7
            bb = self.res_map
            # aa = self.res_map[self.blocks * self.layers-i-1](residual)#随便放过来的记得去掉 这条命令有问题
            filter = self.filter_convs[i](residual)#(16,32,170,29)
            cc = self.skip_map
            dd = self.filter_convs
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)#(16,32,170,29)
            x = filter * gate#(16,32,170,29)

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)#(16,32,170,29)
            try:
                skip = self.skip_map[self.blocks * self.layers-i-1](skip)
            except:
                skip = 0
            skip = s + skip

            # graph conv
            x = self.gconv[i](x, support)
            # residual connect
            aa = self.res_map[self.blocks * self.layers-i-1](residual)#把它放在上面看看
            x = x + self.res_map[self.blocks * self.layers-i-1](residual)

            x = self.bn[i](x)

        x = skip
        x = self.end_conv(x)

        return x

    def decoder(self, inputs):
        support_trans = [torch.eye(self.num_nodes, self.num_nodes).to(self.device)+F.softmax(F.relu(torch.mm(self.nodevec1_trans, self.nodevec2_trans)), dim=1)]

        y = inputs
        y = self.end_conv_trans(y)
        skip = 0
        for i in reversed(range(self.blocks * self.layers)):
            residual = y
            # dilated convolution
            filter = self.filter_convs_trans[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs_trans[i](residual)
            gate = torch.sigmoid(gate)
            y = filter * gate

            # parametrized skip connection
            s = y
            s = self.skip_convs_trans[i](s)
            try:
                skip = self.skip_map_trans[self.blocks * self.layers-i-1](skip)
            except:
                skip = 0
            skip = s + skip

            y = self.gconv_trans[i](y, support_trans)

            # residual connect (trans)
            pad_r = self.res_map_trans[self.blocks * self.layers-i-1](residual)
            y = y + pad_r

            y = self.bn_trans[i](y)

        y = self.start_conv_trans(skip)

        output = y
        out_len = y.size(3)
        if out_len > self.seq_len:
            output = output[:, :, :, -self.seq_len:]

        return output

    def forward(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)
        :return: (batch_size, 1, node_num, predict_length)
        """
        hidden_state = self.encoder(inputs)
        output = self.decoder(hidden_state)
        return output


class stae_predict(nn.Module):#nn.Module是PyTorch中所有神经网络模块的基类。 隐藏状态投影到未来状态的网络
    def __init__(self, train_len=12, num_nodes=207, node_dim=10, horizon=12, device='cuda:0', dropout=0.3,
                 blocks=4, layers=2, kernel_size=2, channels=32, out_dim=2, att_heads=None, load_model=False,
                 model_path=None, fixed=False, map_func='att'):
        super(stae_predict, self).__init__()
        self.train_len = train_len
        self.horizon = horizon
        self.map_func = map_func
        self.dropout = dropout
        self.fc_layers = att_heads
                       
        self.stae = final_ae_stcnn(num_nodes=num_nodes, seq_len=train_len, node_dim=node_dim,
                             blocks=blocks, layers=layers, kernel_size=kernel_size, dropout=dropout,
                             channels=channels, out_dim=out_dim, device=device).to(device)#定义网络

        self.flow = LSTMModel(input_size=4, seq_len=train_len, hidden_size=16,mlp_output_size=4, output_size=4,seq_out_len=12,device='cuda:0').to(device)
        self.gated_final = gated_final(input_size=4, hidden_size=16,mlp_output_size=4, output_size=4,seq_out_len=12,device='cuda:0').to(device)
        if fixed:
            print('------The autoencoder is fixed!--------')
            self.stae.requires_grad_(False)#是将张量标记为不需要计算梯度的标志。这对于需要在模型中冻结某些层的参数时很有用，因为它可以防止这些参数的梯度被计算和更新。

        if load_model:
            checkpoint_train = torch.load(model_path, map_location=device)
            model_dict_enc = self.stae.state_dict()#model.state_dict()是Pytorch中用于查看网络参数的方法。 一般来说，前者多见于优化器的初始化，例如：后者多见于模型的保存
            checkpoint_enc = {k: v for k, v in checkpoint_train.items() if k in model_dict_enc.keys()}
            model_dict_enc.update(checkpoint_enc)
            print(model_dict_enc.keys())
            self.stae.load_state_dict(model_dict_enc)

        self.attention = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.att_heads = att_heads
        if self.map_func == 'att':
            for i in att_heads:
                self.attention.append(gated_selfatt_T(num_node=num_nodes, seq_len=out_dim, num_heads=i, dropout=dropout))
        elif self.map_func == 'fc':
            for i in range(self.fc_layers):
                self.fc.append(nn.Linear((int)(num_nodes * out_dim / 2**i), (int)(num_nodes * out_dim / 2**(i+1))))
            for i in reversed(range(self.fc_layers)):
                self.fc.append(nn.Linear((int)(num_nodes * out_dim / 2**(i+1)), (int)(num_nodes * out_dim / 2**i)))
            for i in range(2*self.fc_layers):
                print("w_size: ", self.fc[i].weight.shape)
                nn.init.eye_(self.fc[i].weight)
                nn.init.constant_(self.fc[i].bias, 0.0)


    def forward(self, inputs):
        """
        :param inputs: (batch_size, dim, node_num, seq_length)(16,1,170,30)  不对 输入是16 1 4 12
        :return: (batch_size, predict_length, node_num, 1)
        """
        history_hidden = self.stae.encoder(inputs)       # [batch_size, out_dim, node_num, 1]

        if self.map_func == 'att':
            predict_hidden = history_hidden.transpose(1,3)
            for i in range(len(self.att_heads)):
                predict_hidden = self.attention[i](predict_hidden)
            predict_hidden = predict_hidden.transpose(1,3)
            # res
            predict_hidden = predict_hidden + history_hidden
        elif self.map_func == 'fc':
            shape = history_hidden.shape
            predict_hidden = history_hidden.flatten(start_dim=1)
            for i in range(2*self.fc_layers):
                predict_hidden = self.fc[i](predict_hidden)
                predict_hidden = F.dropout(predict_hidden, self.dropout, self.training)
            predict_hidden = predict_hidden.reshape(shape)
            predict_hidden = predict_hidden + history_hidden
        elif self.map_func == 'lstm':
            predict_hidden = history_hidden.squeeze().transpose(0,1)
            predict_hidden = self.start_lin(predict_hidden)
            predict_hidden, _ = self.lstm(predict_hidden)
            predict_hidden = self.end_lin(predict_hidden)
            predict_hidden = predict_hidden.transpose(0,1).unsqueeze(-1) 
        predict_output = self.stae.decoder(predict_hidden)
        output = predict_output[:, :, :, -self.horizon:]#16,1,4,12

        #首先16，1，4，12 变成16，4，12   再提取z16，4，11

        inputs_x = inputs.squeeze(1)#16,4,12

        #inputs_x = inputs_x.reshape(inputs_x.shape[0], inputs_x.shape[2], inputs_x.shape[1])#16,12,4

        inputs_z1 = inputs_x[:,:,:self.train_len-1]
        inputs_z2 = inputs_x[:,:,1:]
        inputs_z = inputs_z2 - inputs_z1
        inputs_z = inputs_z.reshape(inputs_z.shape[0], inputs_z.shape[2], inputs_z.shape[1])#16,11,4

        inputs_x = inputs_x.reshape(inputs_x.shape[0], inputs_x.shape[2], inputs_x.shape[1])#16,12,4


        output_flow = self.flow(inputs_x,inputs_z)#16,12,4这里改一下

        output_flow = output_flow.unsqueeze(1)#16,1,12,4
        #这里维数需要变一下，线性网络必须两维，可是这里是3为
        output_flow = output_flow.permute(0,1,3,2)

        output = self.gated_final(output_flow,output)



        return output#16,1,4,output_size
