from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, pre_model: nn.Module, task_idx: int, old_datasetss: list, model_name="GDN"):


        self.pre_model = pre_model
        self.old_dataset = old_datasetss
        self.task_idx = task_idx
        self.model_name = model_name

        self.params = {n: p for n, p in self.pre_model.named_parameters() if p.requires_grad}

        self._precision_matrices = self._diag_fisher()



    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)


        self.pre_model.train()
        for tt in range(self.task_idx):
            for input, target, _, edge_index in self.old_dataset[tt]:
                input, target, edge_index = [item.float() for item in [input, target, edge_index]]
                input, target, edge_index = variable(input), variable(target), Variable(edge_index)
                self.pre_model.zero_grad()
                if self.model_name == "GDN":
                    output = self.pre_model(input, edge_index)
                else:
                    output = self.pre_model(input)

                loss = F.mse_loss(output, target)
                loss.backward()

                for n, p in self.pre_model.named_parameters():
                    precision_matrices[n].data += p.grad.data ** 2 / len(self.old_dataset)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module, pre_model: nn.Module):
        loss = 0
        for (_, param), (old_name, param_old) in zip(model.named_parameters(), pre_model.named_parameters()):
            _loss = self._precision_matrices[old_name] * (param_old - param) ** 2
            loss += _loss.sum()
        return loss





def normal_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, model_name="GDN"):
    model.train()
    epoch_loss = 0
    for input, target, _, edge_index in data_loader:
        input, target, edge_index = [item.float() for item in [input, target, edge_index]]
        input, target, edge_index = variable(input), variable(target), Variable(edge_index)
        optimizer.zero_grad()
        if model_name == "GDN":
            output = model(input, edge_index)
        else:
            output = model(input)
        # print('输出形状')
        output_cpu = output.cpu()
        # print(output_cpu.shape)
        target_cpu = target.cpu()
        # print(target_cpu.shape)
        # outputmle = output[:, 7]
        outputmle = output
        # pdb.set_trace()
        # targetmle = target[:, 7]
        targetmle = target
        loss = F.mse_loss(outputmle, targetmle)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)


def ewc_train(model: nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader,
              ewc: EWC, importance: float, pre_model: nn.Module, model_name="GDN"):
    model.train()
    epoch_loss = 0
    for input, target, _, edge_index in data_loader:
        input, target, edge_index = [item.float() for item in [input, target, edge_index]]
        input, target, edge_index = variable(input), variable(target), Variable(edge_index)
        optimizer.zero_grad()
        if model_name == "GDN":
            output = model(input, edge_index)
        else:
            output = model(input)
        # loss = F.mse_loss(output[:, 7], target[:, 7]) + importance * ewc.penalty(model, pre_model)
        loss = F.mse_loss(output, target) + importance * ewc.penalty(model, pre_model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)
