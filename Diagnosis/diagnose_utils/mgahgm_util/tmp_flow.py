import os

import numpy as np

from args import get_args
import torch

from gat_game import GatGame
from ts_flows import MAF
from utils import get_data, SlidingWindowDataset, get_target_dims, create_data_loaders

if __name__ == "__main__":
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
    args_summary = str(args.__dict__)
    print(args_summary)


    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test), fc_edge_index = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), fc_edge_index = get_data(dataset, normalize=normalize)
    elif dataset == "SWAT":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), fc_edge_index = get_data(dataset, normalize=normalize, max_train_size=1000, max_test_size=1000)
    elif dataset == "WADI":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), fc_edge_index = get_data(dataset, normalize=normalize)
    elif dataset == "PSM":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test), fc_edge_index = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    n_features = x_train.shape[1]  # smap 25

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )


    model = GatGame(
        n_features,  # 特征数k
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        # gru_hid_dim=args.gru_hid_dim,
        # forecast_n_layers=args.fc_n_layers,
        # forecast_hid_dim=args.fc_hid_dim,
        # recon_n_layers=args.recon_n_layers,
        # recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha,
        device=device,
        mask_rate=args.mask_rate,
        # mask_rate=params_ad['mask_rate'],
        replace_rate=args.replace_rate,
        # replace_rate=params_ad['replace_rate'],
        convEn=args.convEn,
        remaskEn=args.remaskEn,
        fc_edge_index=fc_edge_index,
        batch_size=batch_size,
        # topk=int(params_ad['topk'])
        topk=args.topk
    )
    if device == "cuda":
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=weight_decay, eps=args.eps)

    for epoch in range(100):
        model.train()
        l = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # flow_log_prob, vae_log_prob = model(x)
            model(x)
            loss = model.total_loss
            loss.backward()
            l.append(loss.item())
            # if args.clip_grad:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                # torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
        print('epoch ',epoch+1, '----train loss: ', np.mean(l))
        print('flow_loss: ',model.flow_loss.item() , ', vae_loss: ', model.vae_loss.item() )

        # l = []
        # if val_loader is not None:
        #     model.eval()
        #     for x, y in val_loader:
        #         x = x.to(device)
        #         y = y.to(device)
        #         loss = -model(x).mean()
        #         l.append(loss.item())
        # print('val loss: ', np.mean(l))

