from tqdm import tqdm
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE import utils, models,pot
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util import main
from torch.utils.data import DataLoader
import numpy as np
import torch
import time
import torch.nn as nn
import os
import pandas as pd
# dataset='SWaT', model='MCFMAAE', test=False, retrain=False, less=False, epochs=10




def load_dataset(data_dir='./data/mcfmaae_data', dataset='SMD', less=False, percentage=0.2):
    """
    Load the training set, test set, and labels for a specific dataset.
    Args:
        dataset (str): Name of the dataset to load.

    Returns:
        tuple: A tuple containing the training DataLoader, test DataLoader, and labels.
    """
    loader = []
    folder = os.path.join(data_dir, dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"文件夹 {folder} 不存在，已创建。")
    for file in ['train', 'labels']:
        # 获取数据集路径
        if dataset == 'SMD':
            file = 'machine-1-1_' + file
        if dataset == 'SMAP':
            file = 'P-1_' + file
        if dataset == 'MSL':
            file = 'C-1_' + file
        if dataset == 'UCR':
            file = '136_' + file
        if dataset == 'NAB':
            file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    if less:
        loader[0] = utils.cut_array(percentage, loader[0])
    train_loader = DataLoader(
        loader[0], batch_size=loader[0].shape[0])      # SWaT_train: 3000
    labels = loader[1]
    dims = labels.shape[1]
    return train_loader, dims





def load_test_data(data_dir, dataset):
    loader = []
    folder = os.path.join(data_dir, dataset)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"文件夹 {folder} 不存在，已创建。")
    for file in ['test', 'labels']:
        # 获取数据集路径
        if dataset == 'SMD':
            file = 'machine-1-1_' + file
        if dataset == 'SMAP':
            file = 'P-1_' + file
        if dataset == 'MSL':
            file = 'C-1_' + file
        if dataset == 'UCR':
            file = '136_' + file
        if dataset == 'NAB':
            file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    test_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    label_data = loader[1]
    return test_loader, label_data


def create_model(dims, modelname, dataset,epoch):
    model_class = getattr(models, modelname)
    model = model_class(dims, dataset).double()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    # NEW  model
    print(f"{utils.color.GREEN}Creating new model: {modelname}{utils.color.ENDC}")
    accuracy_list = []
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        "accuracy_list": accuracy_list
    }



def load_model(model_dir, modelname, datasetname, dims):
    model_class = getattr(models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'{model_dir}/{modelname}_{datasetname}/model.ckpt'
    # Load the existing model
    print(f"{utils.color.GREEN}Loading pre-trained model: {model.name}{utils.color.ENDC}")
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    accuracy_list = checkpoint['accuracy_list']

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "epoch": epoch,
        "accuracy_list": accuracy_list
    }



def train(train_loader=None, model_dict=None, dataset='data', model_dir='/QUAN/Industrial_time_series_analysis/Diagnose/diagnose_utils/new_mcfmaae_util/checkpoints'):
    # Prepare data
    trainD = next(iter(train_loader))
    trainO = trainD   # Save the original data
    # Get data based on window size
    trainD = convert_to_windows(trainD, model_dict['model'])
    model = model_dict['model']
    optimizer = model_dict['optimizer']
    scheduler = model_dict['scheduler']
    epoch = model_dict['epoch']
    accuracy_list = model_dict['accuracy_list']
    print(
        f'{utils.color.HEADER}Training {model_dict["model"].name} on {dataset}{utils.color.ENDC}')
    # Parameter settings
    num_epochs = epoch
    start = time.time()
    for e in tqdm(list(range(1, num_epochs+1))):     # Show training progress
        # Single training
        lossT, lr = main.backprop(
            e, model, trainD, trainO, optimizer, scheduler)
        accuracy_list.append((lossT, lr))
    # Print total training time, save model, plot accuracy curve
    print(utils.color.BOLD+'Training time: ' +
          "{:10.4f}".format(time.time()-start)+' s'+utils.color.ENDC)
    folder = os.path.expanduser(os.path.join("~", model_dir))
    save_model(model_dir, model, optimizer,
               scheduler, e, accuracy_list, dataset)
    
    return trainO,trainD
   # plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')
    # Plot curves
    # if not args.test:
    #     if 'MCFMAAE' in model.name:
    #         testO = torch.roll(testO, 1, 0)
    #     plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss_test, labels)
  
def test(model, optimizer, scheduler, dataset,test_loader=None, labels=None,trainO=None,trainD=None):
     # Prepare data
    model.eval()
    testD = next(iter(test_loader))
    testO = testD   # Save the original data
    # Get data based on window size
    testD = main.convert_to_windows(testD, model)
    print(f'{utils.color.HEADER}Testing {model.name} {utils.color.ENDC}')
    loss_test, y_pred = main.backprop(
        0, model, testD, testO, optimizer, scheduler, training=False)
        # Scores
    
    df = pd.DataFrame()
    loss_train, _ = main.backprop(0, model, trainD, trainO,
                             optimizer, scheduler, training=False)
    preds = []
    for i in range(loss_test.shape[1]):
        lt, l, ls = loss_train[:, i], loss_test[:, i], labels[:, i]
        result, pred = pot.pot_eval(lt, l, ls,dataset=dataset,modelname=model.name)
        preds.append(pred)
        # print(i)
        df = df._append(result, ignore_index=True)
    lossTfinal, lossFinal = np.mean(
        loss_train, axis=1), np.mean(loss_test, axis=1)
        
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, answer = pot.pot_eval(lossTfinal, lossFinal, labelsFinal,dataset=dataset,modelname=model.name)
    return result,[int(x) for x in answer]


def convert_to_windows(data, model):
    """
    Function to convert input data into windows based on the specified window size.
    Args:
        data: Input data
        model: Model specification containing the window size.

    Returns:
        torch.Tensor: Tensor containing the converted windows.
    """
    w_data = []                 # list of the data with window size
    w_size = model.n_window     # window size
    for i, g in enumerate(data):
        # Ensure sliding window sampling on the dataset
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        w_data.append(w)
    return torch.stack(w_data)


def save_model(model_dir, model, optimizer, scheduler, epoch, accuracy_list, dataset):
    """
    Save the model, optimizer, scheduler, epoch, and accuracy list to a checkpoint file.
    Args:
        model: The model to be saved.
        optimizer: The optimizer used for training the model.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch number.
        accuracy_list: List of accuracy values during training.

    """
    folder = os.path.expanduser(os.path.join("~", model_dir))
    folder = f'{model_dir}/{model.name}_{dataset}'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)
