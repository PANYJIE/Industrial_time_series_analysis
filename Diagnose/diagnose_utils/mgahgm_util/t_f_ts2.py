import torch.optim as optim

from args import get_args
from utils import *
import torch
import ts_flows as fnn
import numpy as np

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

device = "cuda"

(x_train, _), (x_test, y_test), fc_edge_index = get_data(dataset, normalize=normalize)

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


flow = fnn.MAF( n_blocks=num_blocks, input_size=n_features, hidden_size=n_features, n_hidden=4 )
flow.to(device)
optimizer = optim.Adam(flow.parameters(), lr=args.init_lr, weight_decay=1e-6)
'''===========train============'''
cond_data = None
for epoch in range(args.epochs):
    flow.train()
    train_loss = []
    batch_idx = 0

    for x, y in train_loader:
        x = x.to(device)
        x += torch.rand_like(x)

        y = y.to(device)
        optimizer.zero_grad()
        loss = -flow.log_prob(x, cond_data).unsqueeze(-1)  # mean()之前 [256, 100]
        loss_weights, _ = x.min(dim=-1, keepdim=True)

        # assert_shape(loss_weights, (-1, seq_len, 1))
        loss = weighted_average(loss, weights=loss_weights, dim=1)
        loss = loss.mean()
        train_loss.append(loss.item() )

        loss.backward()
        optimizer.step()
        # print('Train, Log likelihood in nats: {:.6f}'.format(-train_loss / (batch_idx + 1) ))
        # batch_idx += 1
    print('epoch: ', epoch, ' loss: ', np.mean(train_loss) )

    '''===========val============'''
    flow.eval()

    val_loss = []
    batch_idx = 0
    for x, y in val_loader:
        x = x.to(device)
        x += torch.rand_like(x)
        y = y.to(device)
        with torch.no_grad():
            loss = -flow.log_prob(x, cond_data).unsqueeze(-1) # sum up batch loss
            loss_weights, _ = x.min(dim=-1, keepdim=True)

            # assert_shape(loss_weights, (-1, seq_len, 1))

            loss = weighted_average(loss, weights=loss_weights, dim=1)
            loss = loss.mean()
            val_loss.append(loss.item())
        # print('Val, Log likelihood in nats: {:.6f}'.format(-val_loss / (x.shape[0]*x.shape[1] ) ))
        # batch_idx += 1
    print('Val loss: ', np.mean(val_loss))



lf = setup_loss_fn(loss_fn=forecast_loss_fn, alpha_l=2)
for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    tx = flow.sample((1, 100))
    # tx = flow.sample(x[x.shape[0]-1].unsqueeze(0).shape)
    real = x[x.shape[0]-1].unsqueeze(0)
    forecast_feat_loss = lf(real , tx)
    print('test: ',forecast_feat_loss)





