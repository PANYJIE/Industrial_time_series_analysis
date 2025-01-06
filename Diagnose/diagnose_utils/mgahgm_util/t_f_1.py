import time

import torch.optim as optim
from tqdm import tqdm

from args import get_args
from utils import *
import torch
import model.flows as fnn

args = get_args()

dataset = args.dataset
window_size = args.lookback
spec_res = args.spec_res
normalize = args.normalize
n_epochs = args.epochs
batch_size = args.bs
init_lr = args.init_lr
val_split = args.val_split
shuffle_dataset = args.shuffle_dataset
use_cuda = args.use_cuda
print_every = args.print_every
log_tensorboard = args.log_tensorboard
group_index = args.group[0]
index = args.group[2:]
forecast_loss_fn = args.forecast_loss_fn
recon_loss_fn = args.recon_loss_fn
weight_decay = args.weight_decay

device = "cpu"

(x_train, _), (x_test, y_test), fc_edge_index = get_data(dataset, normalize=normalize, max_train_size=2000,
                                                         max_test_size=500)

x_train = torch.from_numpy(x_train).float()
x_test = torch.from_numpy(x_test).float()
n_features = x_train.shape[1]
train_dataset = SlidingWindowDataset(x_train, window_size)
test_dataset = SlidingWindowDataset(x_test, window_size)

train_loader, val_loader, test_loader = create_data_loaders(
    train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset)

'''===========flow============'''
act = 'tanh'
num_blocks = 10
modules = []
for _ in range(num_blocks):
    modules += [
        fnn.MADE(n_features, n_features, n_features, act=act),
        # fnn.BatchNormFlow(window_size),
        # fnn.Reverse(n_features)
    ]
flow = fnn.FlowSequential(*modules)

optimizer = optim.Adam(flow.parameters(), lr=args.init_lr, weight_decay=1e-6)
'''===========train============'''
cond_data = None
for epoch in range(args.epochs):
    flow.train()
    train_loss = 0
    batch_idx = 0
    print('epoch: ', epoch)
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = -flow.log_probs(x, cond_data).mean()
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Train, Log likelihood in nats: {:.6f}'.format(-train_loss / (batch_idx + 1)))
        batch_idx += 1

    '''===========val============'''
    flow.eval()
    val_loss = 0
    batch_idx = 0
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            loss = -flow.log_probs(x, cond_data) # sum up batch loss
            val_loss += loss.sum().item()
        print('Val, Log likelihood in nats: {:.6f}'.format(-val_loss / (x.shape[0]*x.shape[1] ) ))
        batch_idx += 1


mode = 'inverse'
lf = setup_loss_fn(loss_fn=forecast_loss_fn, alpha_l=2)
for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    noise = torch.Tensor(1, x.shape[1], x.shape[2]).normal_()
    tx, _ = flow(noise, mode=mode)
    real = x[x.shape[0]-1].unsqueeze(0)
    forecast_feat_loss = lf(real , tx)
    print('test: ',forecast_feat_loss)



    # def sample(self, num_samples=None, noise=None, cond_inputs=None):
    #     if noise is None:
    #         noise = torch.Tensor(num_samples, self.num_inputs).normal_()
    #     device = next(self.parameters()).device
    #     noise = noise.to(device)
    #     if cond_inputs is not None:
    #         cond_inputs = cond_inputs.to(device)
    #     samples = self.forward(noise, cond_inputs, mode='inverse')[0]
    #     return samples