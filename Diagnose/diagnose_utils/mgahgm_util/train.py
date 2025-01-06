import json
from datetime import datetime
import pandas as pd
# import nni
# from args import get_args
# from gat_game import GatGame
# from prediction import Predictor
# from training import Trainer
# from utils import *
import os
import warnings
import numpy as np
import torch
import random


def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)

# if __name__ == "__main__":
#     warnings.filterwarnings('ignore')
#     # torch.autograd.set_detect_anomaly(True)
#     id = datetime.now().strftime("%d%m%Y_%H%M%S")
#     # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
#     # os.environ["CUDA_VISIBLE_DEVICES"] = "PCI_BUS_ID"
#     args = get_args()

#     dataset = args.dataset
#     window_size = args.lookback
#     normalize = args.normalize
#     n_epochs = args.epochs
#     batch_size = args.bs
#     init_lr = args.init_lr
#     val_split = args.val_split
#     shuffle_dataset = args.shuffle_dataset
#     use_cuda = args.use_cuda
#     print_every = args.print_every
#     log_tensorboard = args.log_tensorboard
#     group_index = args.group[0]
#     index = args.group[2:]
#     weight_decay = args.weight_decay

#     args_summary = str(args.__dict__)

#     print(args_summary)
#     params_ad = {'lr': init_lr, 'mask_rate': args.mask_rate, 'replace_rate': args.replace_rate,
#                  'topk': args.topk, 'flow_n_hidden': args.flow_n_hidden, 'flow_n_blocks': args.flow_n_blocks,
#                  'tiaocan': args.tiaocan}
#     '''
# ==============================nni====================================
#     '''
#     # optimized_params = nni.get_next_parameter()
#     # params_ad.update(optimized_params)
#     # print('超参微调:', params_ad)
#     '''
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^nni^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     '''

#     # ---------------------------------------------------#
#     #   设置种子
#     # ---------------------------------------------------#
#     # 定义随机种子固定的函数
#     def get_random_seed(seed):
#         random.seed(seed)
#         os.environ['PYTHONHASHSEED'] = str(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.random.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#         # torch.use_deterministic_algorithms(True)

#     get_random_seed(args.manual_seed)

#     device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"

#     if dataset == 'SMD':
#         output_path = f'output/SMD/{args.group}'
#         (x_train, _), (x_test, y_test) = get_data(
#             f"machine-{group_index}-{index}", normalize=normalize, missing_rate=args.missing_rate)
#     elif dataset in ['MSL', 'SMAP']:
#         output_path = f'output/{dataset}'
#         (x_train, _), (x_test, y_test) = get_data(
#             dataset, normalize=normalize, missing_rate=args.missing_rate)
#     elif dataset == "SWAT":
#         output_path = f'output/{dataset}'
#         (x_train, _), (x_test, y_test) = get_data(
#             dataset, normalize=normalize, missing_rate=args.missing_rate)
#     elif dataset == "WADI":
#         output_path = f'output/{dataset}'
#         (x_train, _), (x_test, y_test) = get_data(
#             dataset, normalize=normalize, missing_rate=args.missing_rate)
#     elif dataset == "PSM":
#         output_path = f'output/{dataset}'
#         (x_train, _), (x_test, y_test) = get_data(
#             dataset, normalize=normalize, missing_rate=args.missing_rate)
#     elif dataset == "MSDS":
#         output_path = f'output/{dataset}'
#         (x_train, _), (x_test, y_test) = get_data(
#             dataset, normalize=normalize, missing_rate=args.missing_rate)
#     else:
#         raise Exception(f'Dataset "{dataset}" not available.')

#     log_dir = f'{output_path}/logs'
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     save_path = f"{output_path}/{id}"

#     x_train = torch.from_numpy(x_train).float()
#     x_test = torch.from_numpy(x_test).float()
#     n_features = x_train.shape[1]  # smap 25

#     target_dims = get_target_dims(dataset)
#     if target_dims is None:
#         out_dim = n_features
#         print(f"Will forecast and reconstruct all {n_features} input features")
#     elif type(target_dims) == int:
#         print(f"Will forecast and reconstruct input feature: {target_dims}")
#         out_dim = 1
#     else:
#         print(f"Will forecast and reconstruct input features: {target_dims}")
#         out_dim = len(target_dims)

#     train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
#     test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

#     train_loader, val_loader, test_loader = create_data_loaders(
#         train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
#     )

#     model = GatGame(
#         n_features,  # 特征数k
#         window_size,
#         out_dim,
#         # kernel_size=args.kernel_size,
#         use_gatv2=args.use_gatv2,
#         time_gat_embed_dim=args.time_gat_embed_dim,
#         dropout=args.dropout,
#         alpha=args.alpha,
#         device=device,
#         # mask_rate=args.mask_rate,
#         mask_rate=params_ad['mask_rate'],
#         # replace_rate=args.replace_rate,
#         replace_rate=params_ad['replace_rate'],
#         convEn=args.convEn,
#         remaskEn=args.remaskEn,
#         batch_size=batch_size,
#         topk=int(params_ad['topk']),
#         flow_hidden_size=args.flow_hidden_size,
#         flow_n_hidden=params_ad['flow_n_hidden'],
#         flow_n_blocks=params_ad['flow_n_blocks'],
#         rel=args.rel
#     )
#     if device == "cuda:0":
#         model.cuda()
#     optimizer = torch.optim.Adam(
#         model.parameters(), lr=args.init_lr, weight_decay=weight_decay, eps=args.eps)

#     print(model)

#     trainer = Trainer(
#         model,
#         optimizer,
#         window_size,
#         n_features,
#         target_dims,
#         n_epochs,
#         batch_size,
#         # init_lr,
#         params_ad['lr'],
#         use_cuda,
#         save_path,
#         log_dir,
#         print_every,
#         log_tensorboard,
#         args_summary,
#         loss_min_save=args.loss_min_save,
#         clip_grad=args.clip_grad
#     )
#     trainer.fit(train_loader, val_loader)

#     plot_losses(trainer.losses, save_path=save_path, plot=False)

#     # Check test loss
#     test_loss = trainer.evaluate(test_loader)
#     print(f"Test flow loss: {test_loss[0]:.5f}")
#     print(f"Test vae loss: {test_loss[1]:.5f}")
#     print(f"Test total loss: {test_loss[2]:.5f}")

#     '''
# ==============================nni====================================
#     '''
#     nni.report_final_result(test_loss[2])
#     '''
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^nni^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     '''
#     # Some suggestions for POT args
#     level_q_dict = {
#         "SMAP": (0.90, 0.005),
#         "MSL": (0.90, 0.001),
#         "SMD-1": (0.9950, 0.001),
#         "SMD-2": (0.9925, 0.001),
#         "SMD-3": (0.9999, 0.001),
#         "SWAT": (0.99, 0.001),
#         "WADI": (0.99, 0.001),
#         "PSM": (0.9950, 0.001),
#         "MSDS": (0.9950, 0.001)
#     }
#     key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
#     level, q = level_q_dict[key]
#     if args.level is not None:
#         level = args.level
#     if args.q is not None:
#         q = args.q

#     # Some suggestions for Epsilon args
#     reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1,
#                       "SMD-3": 1, "WADI": 1, "SWAT": 1, "PSM": 1, "MSDS": 1}
#     key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
#     reg_level = reg_level_dict[key]

#     trainer.load(f"{save_path}/model.pt")
#     prediction_args = {
#         'dataset': dataset,
#         "target_dims": target_dims,
#         'scale_scores': args.scale_scores,
#         "level": level,
#         "q": q,
#         'dynamic_pot': args.dynamic_pot,
#         "use_mov_av": args.use_mov_av,
#         "gamma": args.gamma,
#         "reg_level": reg_level,
#         "save_path": save_path,
#     }
#     best_model = trainer.model
#     predictor = Predictor(
#         best_model,
#         window_size,
#         n_features,
#         prediction_args
#     )

#     label = y_test[window_size:] if y_test is not None else None
#     predictor.predict_anomalies(x_train, x_test, label)

#     # Save config
#     args_path = f"{save_path}/config.txt"
#     with open(args_path, "w") as f:
#         json.dump(args.__dict__, f, indent=2)
