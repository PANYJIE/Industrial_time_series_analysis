from Industrial_time_series_analysis.Decide.decide_utils.il_std_util.evaluation import score_func
import csv
import argparse
import os
from copy import deepcopy
from Industrial_time_series_analysis.Decide.decide_utils.il_std_util.utils import EarlyStopping
from Industrial_time_series_analysis.Decide.decide_utils.il_std_util.data.dataset import RUL_Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from Industrial_time_series_analysis.Decide.decide_utils.il_std_util.model import Encoder
from Industrial_time_series_analysis.Decide.decide_utils.il_std_util import utils


class RUL:

    def __init__(self, dropout=0.3, lr=0.007, sensors=['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21'], sequence_length=30, alpha=0.001,
                 batch_size=128, threshold=125, intermediate_dim=300, latent_dim=2, epochs=2, early_stopping_with_loss=False, save_dir='./checkpoints/IL_STD/FD002/base', dataset='FD004', device='cuda:0', load_model_path='./checkpoints/IL_STD/FD004/base/best_model.pt'):
        self.train_config = {'lr': lr,
                             'sensors': sensors,
                             'sequence_length': sequence_length,
                             'alpha': alpha,
                             'batch_size': batch_size,
                             'threshold': threshold,
                             'intermediate_dim': intermediate_dim,
                             'latent_dim': latent_dim,
                             'epochs': epochs,
                             'early_stopping_with_loss': early_stopping_with_loss,
                             'dropout': dropout

                             }

        self.env_config = {
            'save_dir': save_dir,
            'dataset': dataset,
            'device': device,
            'load_model_path': load_model_path
        }

# 数据集加载
    def load_dataset(self):
        x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(
            self.env_config['dataset'], self.train_config['sensors'], self.train_config['sequence_length'], self.train_config['alpha'], self.train_config['threshold'])
        tr_dataset = RUL_Dataset(x_train, y_train)
        val_dataset = RUL_Dataset(x_val, y_val)
        test_dataset = RUL_Dataset(x_test, y_test)

        # Load Loader
        tr_loader = DataLoader(
            tr_dataset, batch_size=self.train_config['batch_size'], shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=self.train_config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return tr_loader, val_loader, test_loader

    def load_test(self):
        x_test, y_test = utils.get_testdata(self.env_config['dataset'], self.train_config['sensors'],
                                            self.train_config['sequence_length'], self.train_config['alpha'], self.train_config['threshold'])
        test_dataset = RUL_Dataset(x_test, y_test)
        # Load Loader
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 训练

    def train(self, tr_loader, val_loader):
        # Load Dataset
        tr_loader, val_loader, test_loader = self.load_dataset()

        # Load Model
        encoder = Encoder(p=self.train_config['dropout']).to(
            self.env_config['device'])

        # Load Optimizer
        optimizer = torch.optim.Adam(
            list(encoder.parameters()), lr=self.train_config['lr'])

        # Load Early Stopping
        early = EarlyStopping(patience=10)
        # Train
        for epoch in range(self.train_config['epochs']):

            encoder.train()

            tr_loss = 0.
            for tr_x, tr_y in tr_loader:
                tr_x, tr_y = tr_x.to(self.env_config['device']), tr_y.to(
                    self.env_config['device'])
                optimizer.zero_grad()
                out = encoder(tr_x).view(-1)
                rmse_loss = torch.sqrt(F.mse_loss(out, tr_y) + 1e-6)
                loss = rmse_loss
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() / len(tr_loader)

        # Validation
            encoder.eval()
            val_loss = 0.
            val_rmse = 0.
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(self.env_config['device']), val_y.to(
                    self.env_config['device'])

                with torch.no_grad():
                    out = encoder(val_x).view(-1)

                rmse_loss = torch.sqrt(F.mse_loss(out, val_y) + 1e-6)
                loss = rmse_loss

                val_loss += loss / len(val_loader)
                val_rmse += rmse_loss.item() / len(val_loader)

            print('Epoch %d : tr_loss %.2f, val_loss %.2f, val_rmse %.2f' %
                  (epoch, tr_loss, val_loss, val_rmse))
            param_dict = {'encoder': deepcopy(encoder.state_dict())}
        # Early Stopping
            if self.train_config['early_stopping_with_loss']:
                early(val_loss, param_dict)
            else:
                early(val_rmse, param_dict)

            if early.early_stop == True:
                break

        # Save Best Model

        if not os.path.exists(self.env_config['save_dir']):
            os.makedirs(self.env_config['save_dir'])
        torch.save(early.model, os.path.join(
            self.env_config['save_dir'], 'best_model.pt'))

    def test(self,  test_loader=None, save_dir=None):

        encoder = Encoder().to(self.env_config['device'])
        # 加载保存的模型状态
        early = EarlyStopping(patience=10)  # 确保early对象被正确初始化
        early.model = torch.load(
            save_dir+'/best_model.pt')  # 加载整个模型状态
        encoder.load_state_dict(early.model['encoder'])
        encoder.eval()
        test_loss = 0.
        test_rmse = 0.
        test_score = 0.
        test_mse_sum = 0.0  # 初始化MSE总和
        mse_all = 0.0
        all_out_lists = []
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(self.env_config['device']), test_y.to(
                self.env_config['device'])

            with torch.no_grad():
                out = encoder(test_x).view(-1)

            rmse_loss = torch.sqrt(F.mse_loss(out, test_y) + 1e-6)
            mse_a = rmse_loss * rmse_loss
            loss = rmse_loss
            test_loss += loss / len(test_loader)
            mse_all += mse_a.item()
            test_rmse = torch.sqrt(torch.tensor(mse_all / len(test_loader)))
            test_score += score_func(test_y, out)

            all_out_lists.append(out.item())

        out_list = out.tolist()

        # print('Final Result : test loss %.2f, test_rmse %.2f, test_score %.2f' % (
        #     test_loss, test_rmse, test_score))
        # with open(os.path.join(self.env_config['save_dir'], 'result.txt'), 'w') as f:
        #     f.writelines('Final Result : test loss %.2f, test_rmse %.2f,test_score %.2f' % (
        #         test_loss, test_rmse, test_score))

        return all_out_lists,test_rmse, test_score
