from Industrial_time_series_analysis.Forecast.forecast_utils.MSNet_util.data_provider.data_factory import data_provider
from Industrial_time_series_analysis.Forecast.forecast_utils.MSNet_util.exp.exp_basic import Exp_Basic
from Industrial_time_series_analysis.Forecast.forecast_utils.MSNet_util.utils.tools import EarlyStopping, adjust_learning_rate, visual
from Industrial_time_series_analysis.Forecast.forecast_utils.MSNet_util.utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
import argparse
import json
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import csv

warnings.filterwarnings('ignore')




def setting(args):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 1)
    return setting





def load(args, path='Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/parse_json/test_arg.json'):

    args_dict = vars(args)
    with open(path, 'rt') as f:
        args_dict.update(json.load(f))

def save(args, path='test_arg.json'):
    current_dir = os.path.dirname(__file__)
    target_dir = os.path.join(current_dir, 'forecast_utils', 'MSNet_util', 'parse_json')
    full_path = os.path.join(target_dir, path)
    with open(full_path, 'wt') as f:
        json.dump(vars(args), f, indent=4)
    # print(123)
    # print(current_dir)
    # print(123)
    # with open('Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/parse_json/' + path, 'wt') as f:
    #     json.dump(vars(args), f, indent=4)



def parameter_setting():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='MSNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='demo', help='model id')
    parser.add_argument('--model', type=str, required=False, default='MSNet9',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='custom2', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/usstock/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='us_stock.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=31, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=31, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=31, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='relu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')



    # STDNet

    # parser.add_argument('--n_s_hidden', type=int, default=0)
    # parser.add_argument('--n_x_hidden', type=int, default=0)
    parser.add_argument('--n_x', type=int, default=0)
    parser.add_argument('--n_s', type=int, default=0)
    parser.add_argument('--shared_weights', type=bool, default=False)
    parser.add_argument('--stack_types', type=str, default=['time_auto', 'space_cnn_att', 'space_cnn_att'])
    parser.add_argument('--rate1', type=float, default=0.5)
    parser.add_argument('--start_stack_types', type=str, default=3 * ['None'])

    # parser.add_argument('--n_cnn_kernel_size', type=int, default=[3, 3, 3])
    # parser.add_argument('--n_blocks', type=int, default=[3, 3, 3])
    # parser.add_argument('--n_layers', type=int, default=3 * [2])
    # parser.add_argument('--n_mlp_units', type=int, default=3 * [[256, 256]])

    # parser.add_argument('--n_cnn_kernel_size', type=int, default=[3, 3])
    # parser.add_argument('--n_blocks', type=int, default=[2, 2, 2])
    # parser.add_argument('--n_layers', type=int, default=3 * [2])
    # parser.add_argument('--n_mlp_units', type=int, default=3 * [[256, 256]])

    parser.add_argument('--n_cnn_kernel_size', type=int, default=[3, 3, 3])
    parser.add_argument('--n_blocks', type=int, default=[3, 3, 3])
    parser.add_argument('--n_layers', type=int, default=3 * [3])
    parser.add_argument('--n_mlp_units', type=int, default=3 * [[256, 256, 256]])


    parser.add_argument('--cnnatt_hidden_dim', type=int, default=256)
    parser.add_argument('--n_harmonics', type=int, default=5)
    parser.add_argument('--n_polynomials', type=int, default=5)
    parser.add_argument('--batch_normalization', type=bool, default=False)
    parser.add_argument('--weight_decay', type=float, default=1e-3)


    # MSNet
    parser.add_argument('--conv_size_one', type=int, default=4)

    args = parser.parse_args()


    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    save(args, '{}_{}_{}_{}.json'.format(args.model, args.model_id, args.seq_len, args.pred_len))

    return args





class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss(size_average=None, reduce=None, reduction='mean')
        return criterion
    
    def csv_pth(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        ###
        current_dir = os.path.dirname(__file__)
        target_dir1 = os.path.join(current_dir, 'forecast_utils', 'MSNet_util', 'data_loader', 'train_dataset.pth')
        target_dir2 = os.path.join(current_dir, 'forecast_utils', 'MSNet_util', 'data_loader', 'vali_dataset.pth')
        target_dir3 = os.path.join(current_dir, 'forecast_utils', 'MSNet_util', 'data_loader', 'test_dataset.pth')
        torch.save(train_loader.dataset,
                   target_dir1)
        torch.save(vali_loader.dataset,
                   target_dir2)
        torch.save(test_loader.dataset,
                   target_dir3)
        return [target_dir1,
                target_dir2,
                target_dir3]
        ###

        # torch.save(train_loader.dataset, 'Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/data_loader/train_dataset.pth')
        # torch.save(vali_loader.dataset, 'Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/data_loader/vali_dataset.pth')
        # torch.save(test_loader.dataset, 'Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/data_loader/test_dataset.pth')
        # return ['Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/data_loader/train_dataset.pth','Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/data_loader/vali_dataset.pth','Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/data_loader/test_dataset.pth']

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()


                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, data_path):
        # 保存loader数据
        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        # torch.save(train_loader.dataset, 'data_loader/train_dataset.pth')
        # torch.save(vali_loader.dataset, 'data_loader/vali_dataset.pth')
        # torch.save(test_loader.dataset, 'data_loader/test_dataset.pth')

        train_loaded_dataset = torch.load(data_path[0])
        train_data = train_loaded_dataset
        train_loader = DataLoader(train_loaded_dataset, batch_size=self.args.batch_size, shuffle=True)
        print("train", len(train_data))
    

        vali_loaded_dataset = torch.load(data_path[1])
        vali_data = vali_loaded_dataset
        vali_loader = DataLoader(vali_loaded_dataset, batch_size=self.args.batch_size, shuffle=True)
        print("vali", len(vali_data))

        test_loaded_dataset = torch.load(data_path[2])
        test_data = test_loaded_dataset
        test_loader = DataLoader(test_loaded_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)
        print("test", len(test_data))

       
       

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    # print(outputs)
                    # print(batch_y)

                    # print("--------------batch_x.size():",batch_x.size())
                    # print("--------------outputs.size():",outputs.size())
                    # print("--------------batch_y.size()",batch_y.size())
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, data_path, test=0):

        train_loaded_dataset = torch.load(data_path[0])
        vali_loaded_dataset = torch.load(data_path[1])
        test_loaded_dataset = torch.load(data_path[2])

        test_loaded_dataset = train_loaded_dataset + vali_loaded_dataset + test_loaded_dataset
        test_loader = DataLoader(test_loaded_dataset, batch_size=self.args.batch_size, shuffle=False, drop_last=True)




        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        ###
        current_dir = os.path.dirname(__file__)
        target_dir4 = os.path.join(current_dir, 'forecast_utils', 'MSNet_util', 'test_results')
        ###
        folder_path = target_dir4 + setting + '/'
        # folder_path = 'Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 假设 preds 和 trues 是已经通过测试模型生成的 NumPy 数组
        preds1 = np.array(preds)
        trues1 = np.array(trues)

        # 进行 reshape 以展开维度，生成二维数据
        preds_flat = preds1.reshape(-1, preds1.shape[-2])
        trues_flat = trues1.reshape(-1, trues1.shape[-2])

        # 保存路径检查
        folder_path = target_dir4 + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 保存预测值为 CSV 文件
        predictions_file = folder_path + 'predictions.csv'
        with open(predictions_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入预测值
            writer.writerow([f"pred_{i}" for i in range(preds_flat.shape[1])])  # 写入列名
            writer.writerows(preds_flat)  # 写入预测数据

        # 保存真实值为 CSV 文件
        true_values_file = folder_path + 'true_values.csv'
        with open(true_values_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # 写入真实值
            writer.writerow([f"true_{i}" for i in range(trues_flat.shape[1])])  # 写入列名
            writer.writerows(trues_flat)  # 写入真实数据
            print(trues_flat)

        print(f"Predictions saved to {predictions_file}")
        print(f"True values saved to {true_values_file}")



        preds = np.array(preds)
        trues = np.array(trues)
        # print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = target_dir4 + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        ####
        current_dir = os.path.dirname(__file__)
        target_dir55 = os.path.join(current_dir, 'forecast_utils', 'MSNet_util', 'result_long_term_forecast.txt')
        f = open(target_dir55, 'a')
        ####


        # f = open("Industrial_time_series_analysis/Forecast/forecast_utils/MSNet_util/result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return preds_flat

