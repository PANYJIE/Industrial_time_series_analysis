from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from .wwl_oil_model import CNN_GRU_Attention

def Train(feature_set, target_set, feature_set_val, target_set_val, num_per_sample, predict_step, model_save_path, train_epoch, device):
    device = torch.device(device)
    model = CNN_GRU_Attention(input_dim=num_per_sample-predict_step, cnn_out_channels=num_per_sample-predict_step, hidden_size_gru=128,
                    gru_num_layers=2, attention_hidden_size=256, output_dim=predict_step).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss_val = torch.nn.MSELoss(reduction='mean')


    feature_set = feature_set.to(device)
    target_set = target_set.to(device)
    # print(feature_set.shape)
    feature_set_val = feature_set_val.to(device)
    target_set_val = target_set_val.to(device)

    # 将数据加载到DataLoader中
    train_dataset = TensorDataset(feature_set, target_set)
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataset = TensorDataset(feature_set_val, target_set_val)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    epoch_train_loss = []
    epoch_val_loss = []
    for epoch in range(train_epoch):
        train_loss = []
        val_loss = []
        for fea_set, tar_set in train_loader:
            pre = model(fea_set)
            # print(pre.shape)
            # 找出轴承温度
            true = tar_set[:, :, 0]
            # print(true)
            train_temp_loss = loss_fn(pre, true)   # pre和true维度要匹配上
            # print(loss)
            # predict的格式：[训练集数据总数，预测时长]
            optimizer.zero_grad()
            train_temp_loss.backward()
            optimizer.step()
            train_loss.append(train_temp_loss.detach().cpu())
        for fea_set_val, tar_set_val in val_loader:
            # # 验证集Loss
            val_pre = model(fea_set_val)
            # val_pre = torch.squeeze(val_pre)
            #
            val_true = tar_set_val[:, :, 0]
            # val_true = val_true.permute(1, 0)
            val_temp_loss = loss_val(val_pre, val_true)
            # print(val_loss)
            val_loss.append(val_temp_loss.detach().cpu())
        epoch_train_loss.append(np.mean(train_loss))
        epoch_val_loss.append(np.mean(val_loss))
        print('epoch:', epoch)
        print('train loss', np.mean(train_loss))
        print('val loss', np.mean(val_loss))
        # scheduler.step()
    torch.save(model, model_save_path)
    return epoch_train_loss, epoch_val_loss