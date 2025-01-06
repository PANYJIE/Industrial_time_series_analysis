import argparse
import math
import time
import torch
import torch.nn as nn
from . import net
from . import util
from . import trainer
import numpy as np


def evaluate(data, X, Y, model, evaluateL, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        with torch.no_grad():
            output = model(X)
        output = torch.squeeze(output)
        if len(output.shape) == 1:
            output = output.unsqueeze(dim=0)
        if predict is None:
            predict = output
            test = Y
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        # 先mse在l1
        total_loss += evaluateL[0](output * scale, Y * scale).item()
        total_loss_l1 += evaluateL[1](output * scale, Y * scale).item()
        n_samples += (output.size(0) * data.m)

    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)
                   ).mean(axis=0) / (sigma_p * sigma_g)
    correlation = (correlation[index]).mean()
    return predict, rse, rae, correlation


def train(data, X, Y, model, criterion, optim, batch_size, train_config, env_config):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)
        if iter % train_config['step_size'] == 0:
            perm = np.random.permutation(range(train_config['num_nodes']))
        num_sub = int(train_config['num_nodes'] / train_config['num_split'])

        for j in range(train_config['num_split']):
            if j != train_config['num_split'] - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]
            # id = torch.tensor(id).to(device)
            id = torch.tensor(id).type(torch.long).to(env_config['device'])
            tx = X[:, :, id, :]
            ty = Y[:, id]
            output = model(tx, id)
            output = torch.squeeze(output)
            scale = data.scale.expand(output.size(0), data.m)
            scale = scale[:, id]
            loss = criterion(output * scale, ty * scale)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * data.m)
            grad_norm = optim.step()

        if iter % 100 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(
                iter, loss.item()/(output.size(0) * data.m)))
        iter += 1
    return total_loss / n_samples


torch.set_num_threads(3)

# 评估参数也根据


def main(train_config, env_config, Data, model):
    if train_config['L1Loss']:
        criterion = nn.L1Loss(size_average=False).to(env_config['device'])
    else:
        criterion = nn.MSELoss(size_average=False).to(env_config['device'])
    evaluateL2 = nn.MSELoss(size_average=False).to(env_config['device'])
    evaluateL1 = nn.L1Loss(size_average=False).to(env_config['device'])

    best_val = 10000000
    optim = trainer.Optim(
        model.parameters(), train_config['optim'], train_config['lr'], train_config['clip'], lr_decay=train_config['weight_decay']
    )

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training')
        for epoch in range(1, train_config['epochs'] + 1):
            epoch_start_time = time.time()
            train_loss = train(Data, Data.train[0], Data.train[1], model, criterion,
                               optim, train_config['batch_size'], train_config, env_config)
            temp, val_loss, val_rae, val_corr = evaluate(Data, Data.valid[0], Data.valid[1], model,
                                                         [evaluateL2, evaluateL1],
                                                         train_config['batch_size'])
            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr), flush=True)
            # Save the model if the validation loss is the best we've seen so far.

            if val_loss < best_val:
                with open(env_config['load'], 'wb') as f:
                    torch.save(model, f)
                best_val = val_loss
            # 每个epoch测试一次（?设置）
            if epoch % 5 == 0:
                temp, test_acc, test_rae, test_corr = evaluate(Data, Data.test[0], Data.test[1], model, [evaluateL2, evaluateL1],
                                                               train_config['batch_size'])
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(
                    test_acc, test_rae, test_corr), flush=True)
        return env_config['load']
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


# if __name__ == "__main__":
#     vacc = []
#     vrae = []
#     vcorr = []
#     acc = []
#     rae = []
#     corr = []
#     for i in range(10):
#         val_acc, val_rae, val_corr, test_acc, test_rae, test_corr = main()
#         vacc.append(val_acc)
#         vrae.append(val_rae)
#         vcorr.append(val_corr)
#         acc.append(test_acc)
#         rae.append(test_rae)
#         corr.append(test_corr)
#     print('\n\n')
#     print('10 runs average')
#     print('\n\n')
#     print("valid\trse\trae\tcorr")
#     print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae), np.mean(vcorr)))
#     print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae), np.std(vcorr)))
#     print('\n\n')
#     print("test\trse\trae\tcorr")
#     print("mean\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae), np.mean(corr)))
#     print("std\t{:5.4f}\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae), np.std(corr)))
