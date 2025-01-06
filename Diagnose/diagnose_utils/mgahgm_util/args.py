import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()

    # -- Data params ---
    parser.add_argument("--dataset", type=str.upper, default="SMD")
    parser.add_argument("--group", type=str, default="1-1",
                        help="Required for SMD dataset. <group_index>-<index>")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--normalize", type=str2bool, default=True)
    # parser.add_argument("--spec_res", type=str2bool, default=False)

    parser.add_argument("--mask_rate", type=float, default=0.3)
    parser.add_argument("--replace_rate", type=float, default=0)
    parser.add_argument("--missing_rate", type=float, default=0)

    # -- Model params ---
    # 1D conv layer
    # parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--use_gatv2", type=str2bool, default=False)
    # parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    # parser.add_argument("--gru_n_layers", type=int, default=1)
    # parser.add_argument("--gru_hid_dim", type=int, default=150)
    # Forecasting Model
    # parser.add_argument("--fc_n_layers", type=int, default=3)
    # parser.add_argument("--fc_hid_dim", type=int, default=150)
    # Reconstruction Model
    # parser.add_argument("--recon_n_layers", type=int, default=1)
    # parser.add_argument("--recon_hid_dim", type=int, default=150)
    # Other
    parser.add_argument("--alpha", type=float, default=0.2)

    parser.add_argument("--convEn", type=str2bool, default=True)
    parser.add_argument("--remaskEn", type=str2bool, default=True)
    parser.add_argument("--topk", type=int, default=15)

    parser.add_argument("--flow_hidden_size", type=int, default=32)
    parser.add_argument("--flow_n_hidden", type=int, default=1)
    parser.add_argument("--flow_n_blocks", type=int, default=1)

    # --- Train params ---
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--shuffle_dataset", type=str2bool, default=True)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--log_tensorboard", type=str2bool, default=True)
    parser.add_argument("--loss_min_save", type=str2bool, default=False)
    parser.add_argument("--clip_grad", type=str2bool, default=False)

    # parser.add_argument("--forecast_loss_fn", type=str, default="mse")
    # parser.add_argument("--recon_loss_fn", type=str, default="mse")

    # --- Predictor params ---
    parser.add_argument("--scale_scores", type=str2bool, default=False)
    parser.add_argument("--use_mov_av", type=str2bool, default=False)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--level", type=float, default=None)
    parser.add_argument("--q", type=float, default=None)
    parser.add_argument("--dynamic_pot", type=str2bool, default=False)
    parser.add_argument("--score_weight", type=float, default=1)

    # --- Other ---
    parser.add_argument("--comment", type=str, default="")
    parser.add_argument("--tiaocan", type=float, default=0.0)
    parser.add_argument("--manual_seed", type=int, default=42)
    parser.add_argument("--rel", type=str2bool, default=False)
    args = parser.parse_args()
    return args
