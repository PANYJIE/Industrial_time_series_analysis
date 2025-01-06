import torch
from .wwl_power_model import PhyGRU

def Train(feature_set, target_set, feature_set_val, target_set_val, predict_step, model_save_path, train_epoch, device):
    train_loss_his = []
    val_loss_his = []
    device = torch.device(device)

    model = PhyGRU(12, 64, predict_step, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss_val = torch.nn.MSELoss(reduction='mean')

    feature_set = feature_set.to(device)
    target_set = target_set.to(device)

    feature_set_val = feature_set_val.to(device)
    target_set_val = target_set_val.to(device)
    for i in range(train_epoch):
        pre = model(feature_set, target_set)
        # 找出功率
        true = target_set[:, :, 0]
        true = true.permute(1, 0)
        # predict的格式：[训练集数据总数，预测时长]
        loss = loss_fn(pre, true)
        # writer.add_scalar('Loss/train', loss, i)
        print('epoch:', i)
        # print('train loss', loss)
        train_loss_his.append(loss.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证集Loss
        val_pre = model(feature_set_val, target_set_val)
        val_pre = torch.squeeze(val_pre)

        val_true = target_set_val[:, :, 0]
        val_true = val_true.permute(1, 0)

        val_loss = loss_val(val_pre, val_true)
        # print('val loss', val_loss)
        val_loss_his.append(val_loss.detach().cpu())
        torch.save(model, model_save_path)
    return train_loss_his, val_loss_his


