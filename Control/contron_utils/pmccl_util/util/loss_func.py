import torch.nn.functional as F

def mse_loss(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')
    return loss
