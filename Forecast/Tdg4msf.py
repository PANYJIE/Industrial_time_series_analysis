
import numpy as np
import torch.nn as nn
import time
import torch
from Industrial_time_series_analysis.Forecast.forecast_utils.tdg4msf_util.util import DataLoaderS
from Industrial_time_series_analysis.Forecast.forecast_utils.tdg4msf_util.net import TDG4MSF
from Industrial_time_series_analysis.Forecast.forecast_utils.tdg4msf_util.train_single_step import main, evaluate
class TDG4MSF_model:
    def __init__(self, log_interval=2000, optim="adam", L1Loss=True, normalize=2, gcn_true=True, buildA_true=True, num_nodes=8, dropout=0.005, subgraph_size=8, node_dim=40, residual_channels=16, seq_in_len=168, seq_out_len=1, horizon=3, layers=1, batch_size=32, lr=0.0001, weight_decay=0.00001, clip=3, tanhalpha=3, epochs=100, num_split=1, step_size=100, load_model_path="./data/tdg4msf_data/model.pt",  dataset="/home/rl/DATA/nzf/CHENGFEI/QUAN/Industrial_time_series_analysis/Forecast/forecast_utils/tdg4msf_util/data/exchange_rate.txt", device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.train_config = {
            'log_interval': log_interval,
            'optim': optim,
            'L1Loss': L1Loss,
            'normalize': normalize,
            'gcn_true': gcn_true,
            'buildA_true': buildA_true,
            'num_nodes': num_nodes,
            'dropout': dropout,
            'subgraph_size': subgraph_size,
            'node_dim': node_dim,
            'residual_channels': residual_channels,
            'seq_in_len': seq_in_len,
            'seq_out_len': seq_out_len,
            'horizon': horizon,
            'layers': layers,
            'batch_size': batch_size,
            'lr': lr,
            'weight_decay': weight_decay,
            'clip': clip,
            'tanhalpha': tanhalpha,
            'epochs': epochs,
            'num_split': num_split,
            'step_size': step_size,
        }
        self.env_config = {
            'data': dataset,
            'device': device,
            'load': load_model_path
        }
        self.Data = DataLoaderS(dataset, 0.6, 0.2, device,
                                horizon, seq_in_len, normalize)
        model = TDG4MSF(gcn_true, buildA_true, num_nodes,
                        device, dropout=dropout, subgraph_size=subgraph_size,
                        node_dim=node_dim, residual_channels=residual_channels,
                        seq_length=seq_in_len, layers=layers, tanhalpha=tanhalpha, layer_norm_affline=False)
        self.model = model.to(device)

        self.evaluateL = [nn.MSELoss(size_average=False).to(
            self.env_config['device']), nn.L1Loss(size_average=False).to(self.env_config['device'])]

    def print(self):
        print(self.train_config, self.env_config)
        nParams = sum([p.nelement() for p in self.model.parameters()])
        print('Number of model parameters is', nParams, flush=True)
    # def save_label():

    def train(self):
        return main(self.train_config, self.env_config, self.Data, self.model)

    def test(self, model_path=-1, test_path=-1):
        if model_path != -1:
            self.env_config['load'] = model_path
        if test_path != -1:
            self.load_test_data(test_path)
        else:
            self.load_test_data()
         # Load the best saved model.
        if self.env_config['load']:
            with open(self.env_config['load'], 'rb') as f:
                model = torch.load(f)

            Data = self.test_data
           # vtest_acc, vtest_rae, vtest_corr = evaluate(Data, Data.valid[0], Data.valid[1], model, self.evaluateL,
           #                                            self.train_config['batch_size'])
            self.predict, test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, self.evaluateL,
                                                                   self.train_config['batch_size'])
            # print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(
            #     test_acc, test_rae, test_corr))
            return test_acc, test_rae, test_corr,self.predict
            # return vtest_acc, vtest_rae, vtest_corr, test_acc, test_rae, test_corr
        else:
            print("Error: 'load' path in env_config is empty.")
            return None, None, None, None, None, None


    def load_test_data(self, test_path=-1):
        if test_path == -1:
            self.test_data = self.Data
        else:


            self.test_data = DataLoaderS(test_path, 0, 0, self.env_config['device'],
                                         self.train_config['horizon'], self.train_config['seq_in_len'], self.train_config['normalize'])


    def save_label(self, save_path=-1):
        if save_path == -1:
            save_path = self.env_config['save_path']

        np.save(save_path, self.predict)

