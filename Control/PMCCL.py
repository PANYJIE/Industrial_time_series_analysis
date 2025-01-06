# -*- coding:utf-8 -*-
"""
Name：PanYunJie
Date：2024-08-05
"""
import numpy as np
from torch.utils.data import DataLoader, Subset
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util.preprocess import build_loc_net, construct_data
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util import ewc_utils
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util.net_struct import get_feature_map, get_prior_graph_struc
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.datasets.TimeDataset import TimeDataset
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.models import LSTM, LSTMVAE, GRU, Transformer, \
    CNN, RNN, MLP, GDN
import os
import random
from pathlib import Path
import pandas as pd
import json
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.test import testtt
import torch.nn as nn
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util.env import *
from Industrial_time_series_analysis.Control.contron_utils.pmccl_util.util.util_time import *

def train(model=None, save_path='', pre_model=None, config={}, train_dataloader=None, val_dataloader=None, ewc=None):
    model_name = config['model']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=config['decay'])
    train_loss_list = []
    val_loss_list = []
    i = 0
    epoch = config['epoch']
    early_stop_win = 75
    stop_improve_count = 0
    dataloader = train_dataloader
    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        if ewc is None:

            loss = ewc_utils.normal_train(model, optimizer, train_dataloader, model_name)
            print('Normal training')

        else:
            loss = ewc_utils.ewc_train(model, optimizer, train_dataloader, ewc, 5, pre_model, model_name)
            print('Continuous learning training')

        train_loss_list.append(loss)
        acu_loss += loss

        i += 1

        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
            i_epoch, epoch,
            acu_loss / len(dataloader), acu_loss), flush=True
        )


        val_model = model
        val_model.eval()

        if i_epoch > 5:
            if val_dataloader is not None:

                val_loss, val_result = testtt(val_model, val_dataloader, model_name)
                val_loss_list.append(val_loss)
                print(f'epoch {(i_epoch)} Loss:{val_loss}')
                if i_epoch == 6:
                    min_loss = val_loss
                else:
                    if val_loss < min_loss:
                        torch.save(val_model.state_dict(), save_path)
                        min_loss = val_loss
                        stop_improve_count = 0
                    else:
                        stop_improve_count += 1

                    if stop_improve_count >= early_stop_win:
                        break
            else:
                if acu_loss < min_loss:
                    torch.save(val_model.state_dict(), save_path)
                    min_loss = acu_loss


    font1 = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    time_points1 = len(train_loss_list)
    t1 = np.arange(0, time_points1)
    time_points = len(val_loss_list)
    t = np.arange(0, time_points)
    train_loss_list = np.asarray(train_loss_list)
    val_loss_list = np.asfarray(val_loss_list)
    # plt.plot(t, val_loss_list, label='val_loss', color='green')
    # plt.legend(prop=font1, loc='upper center', ncol=2)
    # plt.xlabel('Time (Second)', fontproperties='Times New Roman', fontsize=20)
    # plt.xticks(fontproperties='Arial', fontsize=16)
    # plt.ylabel('Value', fontproperties='Times New Roman', fontsize=20)
    # plt.yticks(fontproperties='Arial', fontsize=16)
    # plt.show()
    # plt.close()
    # plt.plot(t1, train_loss_list, label='train_loss', color='black')
    # plt.legend(prop=font1, loc='upper center', ncol=2)
    # plt.xlabel('Time (Second)', fontproperties='Times New Roman', fontsize=20)
    # plt.xticks(fontproperties='Arial', fontsize=16)
    # plt.ylabel('Value', fontproperties='Times New Roman', fontsize=20)
    # plt.yticks(fontproperties='Arial', fontsize=16)
    # plt.show()
    # plt.close()


    return torch.save(val_model.state_dict(), save_path), train_loss_list,val_loss_list

def test(model, dataloader, model_name='GDN'):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():
            if model_name == 'GDN':
                predicted = model(x, edge_index).float().to(device)
            else:
                predicted = model(x).float().to(device)
            loss = loss_func(predicted[:,7], y[:,7])

            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(test_predicted_list) <= 0:
                test_predicted_list = predicted
                test_ground_list = y
                test_labels_list = labels
            else:
                test_predicted_list = torch.cat((test_predicted_list, predicted), dim=0)
                test_ground_list = torch.cat((test_ground_list, y), dim=0)
                test_labels_list = torch.cat((test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    test_predicted_list = test_predicted_list.tolist()
    test_ground_list = test_ground_list.tolist()
    test_labels_list = test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, test_predicted_list, [test_ground_list, test_labels_list]


def train_data_handle(x_train, list_data, train_config, env_config, adjacency_matrix):
    set_device(env_config['device'])
    train_dataloader_task = []
    val_dataloader_task = []
    train = x_train
    if 'attack' in train.columns:  # only test.csv have 'attack' columns
        train = train.drop(columns=['attack'])
    feature_map = get_feature_map(list_data)
    graph_struc = get_prior_graph_struc(list_data, adjacency_matrix)
    fc_edge_index = build_loc_net(graph_struc, list_data, feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    train_dataset_indata = construct_data(train, feature_map, labels=0)
    cfg = {
        'slide_win': train_config['slide_win'],
        'slide_stride': train_config['slide_stride'],
        'output_step': train_config['output_step'],
    }

    train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
    train_dataloader, val_dataloader = get_loaders(train_dataset,
                                                   train_config['batch'],
                                                   val_ratio=train_config['val_ratio'])


    train_dataloader_task.append(train_dataloader)
    val_dataloader_task.append(val_dataloader)

    train_dataloader_task = [item for sublist in train_dataloader_task for item in sublist]
    val_dataloader_task = [item for sublist in val_dataloader_task for item in sublist]
    return train_dataloader_task, val_dataloader_task


def test_data_handle(x_test, list_data,train_config,adjacency_matrix):
    test_dataset_lst = []
    test_dataset_task = []
    test_dataloader_task = []

    test = x_test
    feature_map = get_feature_map(list_data)
    graph_struc = get_prior_graph_struc(list_data, adjacency_matrix)
    fc_edge_index = build_loc_net(graph_struc, list_data, feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    test_dataset_indata = construct_data(test, feature_map)
    cfg = {
        'slide_win': train_config['slide_win'],
        'slide_stride': train_config['slide_stride'],
        'output_step': train_config['output_step'],
    }
    test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
    test_dataset_task.append(test_dataset)
    test_dataloader_task.append(DataLoader(test_dataset, batch_size=train_config['batch'],
                                           shuffle=False, num_workers=0, drop_last=True))
    test_dataset_task = [item for sublist in test_dataset_task for item in sublist]
    test_dataloader_task = [item for sublist in test_dataloader_task for item in sublist]
    test_dataset_lst.append(test_dataset_task)
    return test_dataloader_task


def get_loaders(train_dataset, batch, val_ratio=0.1):
    dataset_len = int(len(train_dataset))
    train_use_len = int(dataset_len * (1 - val_ratio))
    val_use_len = int(dataset_len * val_ratio)
    val_start_index = random.randrange(train_use_len)
    indices = torch.arange(dataset_len)
    train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
    train_subset = Subset(train_dataset, train_sub_indices)
    val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
    val_subset = Subset(train_dataset, val_sub_indices)
    train_dataloader = DataLoader(train_subset, batch_size=batch,
                                  shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False, drop_last=True)
    return train_dataloader, val_dataloader

def get_model_pratern(list_data, train_config, env_config,adjacency_matrix):
    train_model_task = []
    train = pd.DataFrame([list_data])
    graph_struc = get_prior_graph_struc(list_data, adjacency_matrix)
    feature_map = get_feature_map(list_data)
    fc_edge_index = build_loc_net(graph_struc, list(train.columns), feature_map=feature_map)
    fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
    edge_index_sets = [fc_edge_index]
    if train_config['model'] == "RNN":
        train_model_task.append(RNN.RNNModel(
            dropout_prob=train_config['drop_prob'],
            input_size=len(feature_map),
            hidden_size=train_config['hidden_dim'],
            window_size=train_config['slide_win'],
            num_layers=train_config['num_layers']
        ).to(env_config['device']))

    elif train_config['model'] == "GRU":
        train_model_task.append(GRU.GRUModel(
            dropout_prob=train_config['drop_prob'],
            input_size=len(feature_map),
            hidden_size=train_config['hidden_dim'],
            window_size=train_config['slide_win'],
            num_layers=train_config['num_layers']
        ).to(env_config['device']))


    elif train_config['model'] == "LSTM":
        train_model_task.append(LSTM.LSTModel(
            dropout_prob=train_config['drop_prob'],
            input_size=len(feature_map),
            hidden_size=train_config['hidden_dim'],
            window_size=train_config['slide_win'],
            num_layers=train_config['num_layers']
        ).to(env_config['device']))

    elif train_config['model'] == "LSTMVAE":
        train_model_task.append(LSTMVAE.LSTMVAEModel(
            input_size=len(feature_map),
            hidden_size=train_config['hidden_dim']
        ).to(env_config['device']))


    elif train_config['model'] == "CNN":
        train_model_task.append(CNN.CNNModel(dropout_prob=train_config['drop_prob'],
                                             input_size=len(feature_map),
                                             window_size=train_config['slide_win'],
                                             kernel_length=train_config['kernel_length']
                                             ).to(env_config['device']))

    elif train_config['model'] == "MLP":
        train_model_task.append(MLP.MLPModel(dropout_prob=train_config['drop_prob'],
                                             input_size=len(feature_map),
                                             window_size=train_config['slide_win'],
                                             fc1_size=train_config['fc1_size'],
                                             fc2_size=train_config['fc2_size'],
                                             ).to(env_config['device']))


    elif train_config['model'] == "Transformer":
        train_model_task.append(Transformer.TransformerModel(
            input_size=len(feature_map),
            d_model=train_config['d_model'],
            window_size=train_config['slide_win'],
            num_heads=train_config['num_heads'],
            num_layers=train_config['num_layers'],
            dropout_prob=train_config['drop_prob']
        ).to(env_config['device']))

    elif train_config['model'] == "GDN":
        train_model_task.append(GDN.GDN(edge_index_sets, len(feature_map),
                                        dim=train_config['dim'],
                                        input_dim=train_config['slide_win'],
                                        out_layer_num=train_config['out_layer_num'],
                                        out_layer_inter_dim=train_config['out_layer_inter_dim'],
                                        topk=train_config['topk'],
                                        prior_graph=train_config["graph_struct"],
                                        output_step = train_config["output_step"],
                                        batch = train_config["batch"]
                                        ).to(env_config['device']))
    return train_model_task

def parameter_setting(graph_struct=0, model="GDN", use_ewc=0, batch=32, epoch=200, slide_win=15, slide_stride=1,output_step=5,save_path_pattern="",
           dataset='steel', device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), comment="", dim=32, out_layer_num=1, out_layer_inter_dim= 32,topk=4,
           drop_prob=0.1,hidden_dim=64, num_heads=2, num_layers=1, d_model=8, kernel_length=3, fc1_size=512,
           fc2_size=256,decay=0,val_ratio=0.2,sensor_num=7,report='best',load_model_path=""):
    train_config = {
        'graph_struct': graph_struct,
        'model': model,
        'use_ewc': use_ewc,
        'batch': batch,
        'epoch': epoch,
        'slide_win': slide_win,
        'slide_stride': slide_stride,
        'output_step': output_step,
        'comment': comment,
        'decay': decay,
        'val_ratio': val_ratio,
        'dim': dim,
        'out_layer_num': out_layer_num,
        'out_layer_inter_dim': out_layer_inter_dim,
        'topk': topk,
        'drop_prob': drop_prob,
        'hidden_dim': hidden_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_model': d_model,
        'sensor_num': sensor_num,
        'kernel_length': kernel_length,
        'fc1_size': fc1_size,
        'fc2_size': fc2_size
    }

    env_config = {
        'save_path': save_path_pattern,
        'dataset': dataset,
        'report': report,
        'device': device,
        'load_model_path': load_model_path
    }


    return train_config,  env_config


def get_save_path(model_parameters_save_path, train_config, env_config, task_idx=0):

    dir_path = env_config['save_path']
    dataset = env_config['dataset']
    model = train_config['model']
    datestr = None

    if datestr is None:
        now = datetime.now()
        datestr = now.strftime('%m|%d-%H:%M:%S')

    paths = [
        f'{model_parameters_save_path}/{dir_path}/{dataset}/Task{task_idx}/{model}_task{task_idx}.pt',
        f'{model_parameters_save_path}{dir_path}/{dataset}/Task{task_idx}/{model}_task{task_idx}.csv',
        f'{model_parameters_save_path}{dir_path}/{dataset}/Task{task_idx}/{model}_task{task_idx}.png',
    ]

    for path in paths:
        dirname = os.path.dirname(path)
        Path(dirname).mkdir(parents=True, exist_ok=True)

    return paths



def save_list(data, env_config, task):
    dir_path = env_config['save_path']
    dataset = env_config['dataset']
    filename = f'./pretrained/{dir_path}/{dataset}/{task}.json'
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_list(env_config, task):
    dir_path = env_config['save_path']
    dataset = env_config['dataset']
    filename = f'./pretrained/{dir_path}/{dataset}/{task}.json'
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
