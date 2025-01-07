import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.models import *
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.constants import *
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.utils import *
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.plotting import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
import Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.models
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.pot import *
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mcfmaae_util.MCFMAAE.diagnosis import *


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


def load_dataset(dataset):
    """
    Load the training set, test set, and labels for a specific dataset.
    Args:
        dataset (str): Name of the dataset to load.

    Returns:
        tuple: A tuple containing the training DataLoader, test DataLoader, and labels.
    """
    folder = os.path.join(output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    loader = []
    for file in ['train', 'test', 'labels']:
        # Get the dataset path
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
    if args.less:
        loader[0] = cut_array(0.2, loader[0])
    # Load training, test data, and labels
    train_loader = DataLoader(
        loader[0], batch_size=loader[0].shape[0])      # SWaT_train: 3000
    test_loader = DataLoader(
        loader[1], batch_size=loader[1].shape[0])       # SWaT_test:  5000
    labels = loader[2]
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    """
    Save the model, optimizer, scheduler, epoch, and accuracy list to a checkpoint file.
    Args:
        model: The model to be saved.
        optimizer: The optimizer used for training the model.
        scheduler: The learning rate scheduler.
        epoch (int): Current epoch number.
        accuracy_list: List of accuracy values during training.

    """
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims):
    """
    Load a model with the specified name and dimensions.

    Args:
        modelname (str): Name of the model to load.
        dims (int): Number of dimensions for the model.

    Returns:
        tuple: A tuple containing the loaded model, optimizer, scheduler, epoch, and accuracy list.
    """
    model_class = getattr(MCFMAAE.models, modelname)
    model = model_class(dims).double()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    # Model storage path
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        # Load the existing model
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        # Retrain the model
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    """
        Perform backpropagation for a given epoch on the model using the provided data.
        Args:
            epoch (int): Current epoch number.
            model: The model to perform backpropagation on.
            data: Input data with the window size for the model.
            dataO: Original input data.
            optimizer: Optimizer for updating model parameters.
            scheduler: Learning rate scheduler.
            training (bool): Flag indicating whether the model is in training mode.

        Returns:
            tuple: A tuple containing the average loss and the current learning rate.
        """
    feats = dataO.shape[1]            # Feature dimensions
    window_size = model.n_window       # Window size
    discriminator = DataDiscriminator(
        window_size, feats, hidden_size=256).double()  # Initialize the discriminator
    # Initialize the discriminator model optimizer
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=model.lr)

    # Data loading
    data_double = torch.DoubleTensor(data)
    dataset = TensorDataset(data_double, data_double)
    batch_size = model.batch if training else len(data)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Define losses
    loss_mse = nn.MSELoss(reduction='none')
    adversarial_criterion = nn.BCEWithLogitsLoss()
    ls_total = []       # Store loss values

    # Training
    if training:
        for data, _ in dataloader:
            data_input = data.permute(1, 0, 2)

            # Generation
            data_output = model(data_input)

            # Discrimination
            data_dis_input = data_output.permute(0, 2, 1)
            # Discrimination label
            dis_label = discriminator(data_dis_input.double())

            # Calculate loss: Discrimination loss + Reconstruction loss
            # Generate real label
            label_real = torch.ones(dis_label.shape[0], 1)
            loss_dis = adversarial_criterion(
                dis_label, label_real)  # Discrimination loss
            loss_rec = adversarial_criterion(
                data_output, data_input).mean()     # Reconstruction loss
            loss_total = loss_rec + loss_dis
            ls_total.append(loss_total.item())

            # Optimization
            optimizer.zero_grad()
            optimizer_dis.zero_grad()
            loss_rec.backward(retain_graph=True)
            loss_dis.backward(retain_graph=True)
            optimizer.step()
            optimizer_dis.step()
        scheduler.step()

        # Save the model
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(ls_total)}')
        return np.mean(ls_total), optimizer.param_groups[0]['lr']
    else:
        # Testing
        for data, _ in dataloader:
            local_bs = data.shape[0]
            data_input = data.permute(1, 0, 2)
            elem = data_input[-1, :, :].view(1, local_bs, feats)
            data_output = model(data_input)
            if isinstance(data_output, tuple):
                data_output = data_output[1]
        loss = loss_mse(data_output, elem)[0]
        return loss.detach().numpy(), data_output.detach().numpy()[0]


if __name__ == '__main__':
    # Load the dataset
    train_loader, test_loader, labels = load_dataset(args.dataset)

    # Load the model
    model, optimizer, scheduler, epoch, accuracy_list = load_model(
        args.model, labels.shape[1])

    # Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD    # Save the original data
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(
        testD, model)  # Get data based on window size

    # Training phase
    if not args.test:
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        # Parameter settings
        num_epochs = args.epochs
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):     # Show training progress
            # Single training
            lossT, lr = backprop(e, model, trainD, trainO,
                                 optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        # Print total training time, save model, plot accuracy curve
        print(color.BOLD+'Training time: ' +
              "{:10.4f}".format(time()-start)+' s'+color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list)
        plot_accuracies(accuracy_list, f'{args.model}_{args.dataset}')

    # Testing phase
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')
    loss_test, y_pred = backprop(
        0, model, testD, testO, optimizer, scheduler, training=False)

    # Plot curves
    if not args.test:
        if 'MCFMAAE' in model.name:
            testO = torch.roll(testO, 1, 0)
        plotter(f'{args.model}_{args.dataset}',
                testO, y_pred, loss_test, labels)

    # Scores
    df = pd.DataFrame()
    loss_train, _ = backprop(0, model, trainD, trainO,
                             optimizer, scheduler, training=False)
    for i in range(loss_test.shape[1]):
        lt, l, ls = loss_train[:, i], loss_test[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls)
        preds.append(pred)
        df = df._append(result, ignore_index=True)
    lossTfinal, lossFinal = np.mean(
        loss_train, axis=1), np.mean(loss_test, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal)
    result.update(hit_att(loss_test, labels))
    result.update(ndcg(loss_test, labels))
    print(df)
    pprint(result)
