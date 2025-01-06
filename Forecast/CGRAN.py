from .forecast_utils.cgran_util.wwl_oil_train import Train
from .forecast_utils.cgran_util.wwl_oil_test import Test
import torch

def train(train_set_fea, train_set_tar, val_set_fea, val_set_tar, data_num_per_sample, data_predict_step, model_save_path, train_epoch, device):
    train_set_fea = torch.tensor(train_set_fea, dtype=torch.float32)
    train_set_tar = torch.tensor(train_set_tar, dtype=torch.float32)
    val_set_fea = torch.tensor(val_set_fea, dtype=torch.float32)
    val_set_tar = torch.tensor(val_set_tar, dtype=torch.float32)
    Train(train_set_fea, train_set_tar, val_set_fea, val_set_tar, data_num_per_sample,
          data_predict_step, model_save_path, train_epoch, device)

def test(test_set_fea, test_set_tar, model_save_path, device):
    test_set_fea = torch.tensor(test_set_fea, dtype=torch.float32)
    test_set_tar = torch.tensor(test_set_tar, dtype=torch.float32)
    predict, true = Test(test_set_fea, test_set_tar, model_save_path,
                         device)
    return predict, true







