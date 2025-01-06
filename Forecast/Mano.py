import torch
import torch.nn as nn
import h5py
from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.utils import Train_TransformerOperatorDataset,Test_TransformerOperatorDataset
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import shutil

from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.models.oformer import Encoder1D, STDecoder1D, OFormer1D, PointWiseDecoder1D
from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.models.fno import FNO1d
# from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.models.deeponet import DeepONet1D
import sys

from Industrial_time_series_analysis.Forecast.forecast_utils.mano_util.models.MANO import MANO1D

device = 'cuda' if(torch.cuda.is_available()) else 'cpu'

def get_train_data(f, config):
    print(">>>>>>>>>>>加载训练数据>>>>>>>>>>>")
    train_data = Train_TransformerOperatorDataset(f, config['flnm'],
                            split="train",
                            initial_step=config['initial_step'],
                            reduced_resolution=config['reduced_resolution'],
                            reduced_resolution_t=config['reduced_resolution_t'],
                            reduced_batch=config['reduced_batch'],
                            saved_folder=config['base_path'],
                            return_text=config['return_text'],
                            num_t=config['num_t'],
                            num_x=config['num_x'],
                            sim_time=config['sim_time'],
                            num_samples=config['num_samples'],
                            train_style=config['train_style'],
                            rollout_length=config['rollout_length'],
                            seed=config['seed'],
    )
    train_data.data = train_data.data.to(device)
    train_data.grid = train_data.grid.to(device)
    val_data = Train_TransformerOperatorDataset(f, config['flnm'],
                            split="val",
                            initial_step=config['initial_step'],
                            reduced_resolution=config['reduced_resolution'],
                            reduced_resolution_t=config['reduced_resolution_t'],
                            reduced_batch=config['reduced_batch'],
                            saved_folder=config['base_path'],
                            return_text=config['return_text'],
                            num_t=config['num_t'],
                            num_x=config['num_x'],
                            sim_time=config['sim_time'],
                            num_samples=config['num_samples'],
                            train_style=config['train_style'],
                            rollout_length=config['rollout_length'],
                            seed=config['seed'],
    )
    val_data.data = val_data.data.to(device)
    val_data.grid = val_data.grid.to(device)


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True,
                                               generator=torch.Generator(device='cpu'))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cpu'))


    # Check against data leaks
    assert not (bool(set(train_data.data_list) & \
                     set(val_data.data_list)))


    return train_loader, val_loader


def get_test_data(f, config):
    print(">>>>>>>>>>>加载测试数据>>>>>>>>>>>")
    test_data = Test_TransformerOperatorDataset(f, config['flnm'],
                            split="test",
                            initial_step=config['initial_step'],
                            reduced_resolution=config['reduced_resolution'],
                            reduced_resolution_t=config['reduced_resolution_t'],
                            reduced_batch=config['reduced_batch'],
                            saved_folder=config['base_path'],
                            return_text=config['return_text'],
                            num_t=config['num_t'],
                            num_x=config['num_x'],
                            sim_time=config['sim_time'],
                            num_samples=config['num_samples'],
                            train_style=config['train_style'],
                            rollout_length=config['rollout_length'],
                            seed=config['seed'],
    )
    test_data.data = test_data.data.to(device)
    test_data.grid = test_data.grid.to(device)


    test_loader = torch.utils.data.DataLoader(test_data, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'], shuffle=False,
                                             generator=torch.Generator(device='cpu'))


    return test_loader


def train(config,train_loader, val_loader,model_path,f):
    # print(f'Epochs = {config[\'epochs\']}, learning rate = {config[\'learning_rate\']}, scheduler step = {scheduler_step}    , scheduler gamma = {scheduler_gamma}')
    print(">>>>>>>>>>>开始训练>>>>>>>>>>")
    ################################################################
    # load data
    ################################################################
    # model_name = config['flnm'] + '_{}'.format(config['model']) + ".pt"
    # model_path = path + "/" + model_name

    print("Filename: {}, Seed: {}\n".format(config['flnm'], config['seed']))

    # train_loader, val_loader, test_loader = get_data(f, config)

    neural_operator = get_neural_operator(config['neural_operator'], config)
    # transformer = get_transformer(config['transformer'], neural_operator, config)
    transformer = MANO1D(10, config['hidden'], config['num_x'], augment_dim=config['augment_dim'],
                         dropout=config['dropout'], kernels=config['kernels'], rep_dim=128,
                         neural_operator=neural_operator).to(device=device)
    # print(transformer)
    total_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f'Total parameters = {total_params}')

    # Use Adam as the optimizer.
    print("\nWEIGHT DECAY: {}\n".format(config['weight_decay']))
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['learning_rate'],  # div_factor=1e6,
                                                    steps_per_epoch=len(train_loader), epochs=config['epochs'])

    # Use mean squared error as the loss function.
    # loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = nn.L1Loss(reduction='mean')

    # Train the transformer for the specified number of epochs.
    train_losses = []
    val_losses = []
    loss_val_min = np.infty
    lrs = []
    shift = 0
    for epoch in tqdm(range(config['epochs'])):
        # Iterate over the training dataset.
        train_loss = 0
        times = []
        max_val = 0
        # for x, y, grid, tokens in tqdm(train_loader):
        transformer.train()
        for bn, (x0, y, grid, tokens, t) in enumerate(train_loader):

            start = time.time()
            y_pred = transformer(grid.to(device=device), tokens.to(device=device), x0.to(device=device),
                                 t.to(device=device))
            y = y[..., 0].to(device=device)  # .cuda()

            # Compute the loss.
            loss = loss_fn(y_pred, y)

            # Backward pass: compute gradient of the loss with respect to model
            # parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.state_dict()['param_groups'][0]['lr'])

            train_loss += loss.item()
            if (bn == 0):
                y_train_true = y.clone()
                y_train_pred = y_pred.clone()

            scheduler.step()

            if (bn % 1000 == 0 and len(train_loader) >= 1000):
                print("Batch: {0}\tloss = {1:.4f}".format(bn, train_loss / (bn + 1)))

        # scheduler.step()

        train_loss /= (bn + 1)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_model = transformer
            val_model.eval()
            val_loss = 0
            all_val_preds = []
            for bn, (x0, y, grid, tokens, t) in enumerate(val_loader):
                # Forward pass: compute predictions by passing the input sequence
                # through the transformer.
                y_pred = transformer(grid.to(device=device), tokens.to(device=device), x0.to(device=device),
                                     t.to(device=device))
                y = y[..., 0].to(device=device)  # .cuda()
                if (bn == 0):
                    y_val_true = y.clone()
                    y_val_pred = y_pred.clone()
                all_val_preds.append(y_pred.detach())

                # Compute the loss.
                val_loss += loss_fn(y_pred, y).item()

            if  val_loss < loss_val_min:
                loss_val_min = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': transformer.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_val_min
                    }, model_path)

        val_loss /= (bn + 1)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}: loss = {train_loss:.4f}\t val loss = {val_loss:.4f}")

    return  transformer,train_losses, val_losses



## test
def test(model,test_loader, loss_fn, config):
    # src_mask = generate_square_subsequent_mask(640).cuda()
    print(">>>>>>>>>>>开始测试>>>>>>>>>>>")
    y_preds = []
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for bn, (x0, y, grid, tokens, t) in enumerate(test_loader):
            y_pred = model(grid.to(device=device), tokens.to(device=device), x0.to(device=device),
                                 t.to(device=device))
            y_preds.append(y_pred)
            y = y[..., 0].to(device=device)
            # Compute the loss.
            test_loss += loss_fn(y_pred, y).item()

    test_value = test_loss / (bn + 1)
    test_vals = []
    # test_value = test(test_loader, transformer, loss_fn)
    test_vals.append(test_value)
    # print("TEST VALUE FROM LAST EPOCH: {0:5f}".format(test_value))
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    # # test_value = test(test_loader, transformer, loss_fn)
    # test_vals.append(test_value)
    # print("TEST VALUE BEST LAST EPOCH: {0:5f}".format(test_value))
    # np.save("{}{}__{}/test_vals.npy".format(config['results_dir'], config['model'],
    #                                         config['neural_operator']), test_vals)
    avg_loss = sum(test_vals) / len(test_vals)
    # print("avg MAE loss :{}".format(avg_loss))

    return y_preds[-1], avg_loss


def parameter_setting(base_path= "",results_dir= "",data_name= "",flnm="",train_style='next_step',model= 'mano',neural_operator= 'fno' ,embedding= 'standard',
    continue_training= False,forcing= False,rollout_length= 1,num_workers= 0,batch_size= 128,initial_step= 10,seed=1,
    t_train= 200,model_update= 1,fno= False,return_text= True,reduced_resolution= 1,reduced_resolution_t= 1,reduced_batch= 1,
    epochs= 200,num_seeds= 0,learning_rate= 1.e-3,weight_decay= 1.e-5,scheduler_step= 200,scheduler_gamma= 0.1, training_type= 'single',
    num_x= 100,num_t= 100,sim_time=100, num_samples=100,hidden= 64,layers= 1,heads= 1,dropout= 0.05,log_freq= 1,progress_plot_freq= 10,augment_dim=64,
     width= 64, modes=4,num_channels= 1,kernels= [1,2,4,8],activation= 'silu'):

    train_config = {

        'base_path': base_path,
        'results_dir': results_dir,
        'data_name': data_name,
        'flnm': flnm,
        'augment_dim': augment_dim,

        'train_style': train_style,
        'model': model,
        'neural_operator': neural_operator,  # fno, oformer, deeponet,
        'embedding': embedding,
        'continue_training': continue_training,
        'forcing': forcing,
        'rollout_length': rollout_length,
        'num_workers': num_workers,

        'batch_size': batch_size,
        'initial_step': initial_step,
        't_train': t_train,
        'model_update': model_update,


        'fno': fno,
        'return_text': return_text,
        'reduced_resolution': reduced_resolution,
        'reduced_resolution_t': reduced_resolution_t,
        'reduced_batch': reduced_batch,
        'epochs': epochs,
        'num_seeds': num_seeds,
        # Optimizer
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'scheduler_step': scheduler_step,
        'scheduler_gamma': scheduler_gamma,
        'training_type': training_type,

        #
        'num_x': num_x,
        'num_t': num_t,
        'sim_time': sim_time,
        'num_samples': num_samples,

        # Transformer
        'hidden': hidden,
        'layers': layers,
        'heads': heads,
        'dropout': dropout,
        'seed': seed,
        # Tracking
        'log_freq': log_freq,
        'progress_plot_freq': progress_plot_freq,

        'kernels': kernels,
        # # DeepONet params
        # ###
        # 'branch_net': branch_net,
        # 'trunk_net': trunk_net,
        # FNO params
        ###
        'width': width,
        'modes': modes,
        'num_channels': num_channels,
        'activation': activation,


    }
    prefix = train_config['flnm']+ "_" + train_config['train_style']

    train_config['prefix'] = prefix

    return train_config



def custom_collate(batch):
    x0 = torch.empty((len(batch), batch[0][0].shape[0]))
    y = torch.empty((len(batch), batch[0][1].shape[0], 1))
    grid = torch.empty((len(batch), batch[0][2].shape[0]))
    tokens = torch.empty((len(batch), batch[0][3].shape[0]))
    forcing = []
    time = torch.empty(len(batch))
    for idx, b in enumerate(batch):
        x0[idx] = b[0]
        y[idx] = b[1]
        grid[idx] = b[2]
        tokens[idx] = b[3]
        forcing.append(b[4])
        time[idx] = b[5]
    return x0, y, grid, tokens, forcing, time


def progress_plots(ep, y_train_true, y_train_pred, y_val_true, y_val_pred, path="progress_plots", seed=None):
    ncols = 8
    fig, ax = plt.subplots(ncols=ncols, nrows=2, figsize=(5*ncols,14))
    for i in range(ncols):
        ax[0][i].plot(y_train_true[i].reshape(100,).detach().cpu())
        ax[0][i].plot(y_train_pred[i].reshape(100,).detach().cpu())
        ax[1][i].plot(y_val_true[i].reshape(100,).detach().cpu())
        ax[1][i].plot(y_val_pred[i].reshape(100,).detach().cpu())

    fname = str(ep)
    while(len(fname) < 8):
        fname = '0' + fname
    if(seed is not None):
        plt.savefig("./{}/{}_{}.pdf".format(path, seed, fname))
    else:
        plt.savefig("./{}/{}.pdf".format(path, fname))
    plt.close()


def val_plots(ep, val_loader, preds, path="progress_plots", seed=None):
    im_num = 0
    for vals in val_loader:
        for idx, v in tqdm(enumerate(vals[1])):

            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(v.reshape(100,).detach().cpu())
            ax.plot(preds[0][idx].detach().cpu())
            fname = str(im_num)
            while(len(fname) < 8):
                fname = '0' + fname
            ax.set_title(fname)
            plt.savefig("./val_2/{}_{}.png".format(seed, fname))
            plt.close()

            im_num += 1


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


def get_neural_operator(model_name, config):
    if(model_name == "fno"):
        neural_operator = FNO1d(config['num_channels'], config['modes'], config['width'], config['initial_step'], config['dropout'])
    elif(model_name == "deeponet"):
        neural_operator = DeepONet1D(layer_sizes_branch=config['branch_net'], layer_sizes_trunk=config['trunk_net'],
                                 activation=config['activation'],
                                 kernel_initializer=config['kernel_initializer'])
    elif(model_name == "oformer"):
        encoder = Encoder1D(input_channels=config['input_channels'], in_emb_dim=config['in_emb_dim'],
                            out_seq_emb_dim=config['out_seq_emb_dim'], depth=config['depth'], dropout=config['dropout'],
                            res=config['enc_res'])
        decoder = PointWiseDecoder1D(latent_channels=config['latent_channels'], out_channels=config['out_channels'],
                                     decoding_depth=config['decoding_depth'], scale=config['scale'], res=config['dec_res'])
        neural_operator = OFormer1D(encoder, decoder)
    
    neural_operator.to(device)
    return neural_operator







