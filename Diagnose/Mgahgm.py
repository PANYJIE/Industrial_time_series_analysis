from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util import utils
import torch
import os   
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.training import Trainer
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.gat_game import GatGame
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.utils import get_target_dims,create_data_loaders,SlidingWindowDataset
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.prediction import Predictor
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.train import get_random_seed
import itertools
import warnings
warnings.filterwarnings('ignore')
def params_config(
    dataset="SMD",
    group="1-1",
    lookback=100,
    normalize=True,
    mask_rate=0.3,
    replace_rate=0.0,
    missing_rate=0.0,
    use_gatv2=False,
    time_gat_embed_dim=None,
    alpha=0.2,
    convEn=True,
    remaskEn=True,
    topk=15,
    flow_hidden_size=32,
    flow_n_hidden=1,
    flow_n_blocks=1,
    epochs=30,
    val_split=0.1,
    bs=256,
    init_lr=1e-3,
    shuffle_dataset=True,
    dropout=0.3,
    weight_decay=0.0,
    eps=0.0,
    use_cuda=True,
    print_every=1,
    log_tensorboard=True,
    loss_min_save=False,
    clip_grad=False,
    scale_scores=False,
    use_mov_av=False,
    gamma=1,
    level=None,
    q=None,
    dynamic_pot=False,
    score_weight=1,
    comment="",
    tiaocan=0.0,
    manual_seed=42,
    rel=False,
    data_dir='./dataset/mgahgm_data',
    model_dir='./Industrial_time_series_analysis/Diagnose/diagnose_utils/mgahgm_util/checkpoints',


):
    device = "cuda:0" if use_cuda and torch.cuda.is_available() else "cpu"
    get_random_seed(manual_seed)
    train_config = {

        "group": group,
        "window_size": lookback,
        "normalize": normalize,
        "mask_rate": mask_rate,
        "replace_rate": replace_rate,
        "missing_rate": missing_rate,
        "use_gatv2": use_gatv2,
        "time_gat_embed_dim": time_gat_embed_dim,
        "alpha": alpha,
        "convEn": convEn,
        "remaskEn": remaskEn,
        "topk": topk,
        "flow_hidden_size": flow_hidden_size,
        "flow_n_hidden": flow_n_hidden,
        "flow_n_blocks": flow_n_blocks,
        "epochs": epochs,
        "val_split": val_split,
        "batch_size": bs,
        "init_lr": init_lr,
        "shuffle_dataset": shuffle_dataset,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "eps": eps,
        "print_every": print_every,
        "log_tensorboard": log_tensorboard,
        "loss_min_save": loss_min_save,
        "clip_grad": clip_grad,
        "scale_scores": scale_scores,
        "use_mov_av": use_mov_av,
        "gamma": gamma,
        "level": level,
        "q": q,
        "dynamic_pot": dynamic_pot,
        "score_weight": score_weight,
        "tiaocan": tiaocan,
        "manual_seed": manual_seed,
        "rel": rel,
    }
    env_config = {
        "dataset": dataset,
        "device": device,
        "comment": comment,
        "use_cuda": use_cuda,
        "data_dir": data_dir,
        "model_dir": model_dir,
    }
    return train_config, env_config


def load_data(dataset, train_config):
    group_index = train_config["group"][0]
    index = train_config["group"][2:]
    group = train_config["group"]
    dataset = dataset
    normalize = train_config["normalize"]
    missing_rate = train_config["missing_rate"]
    if dataset == 'SMD':
        output_path = f'output/SMD/{group}'
        (x_train, _), (x_test, y_test) = utils.get_data(
            f"machine-{group_index}-{index}", normalize=train_config['normalize'], missing_rate=train_config['missing_rate'])
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = utils.get_data(
            dataset, normalize=normalize, missing_rate=missing_rate)
    elif dataset == "SWAT":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = utils.get_data(
            dataset, normalize=normalize, missing_rate=missing_rate)
    elif dataset == "WADI":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = utils.get_data(
            dataset, normalize=normalize, missing_rate=missing_rate)
    elif dataset == "PSM":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = utils.get_data(
            dataset, normalize=normalize, missing_rate=missing_rate)
    elif dataset == "MSDS":
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = utils.get_data(
            dataset, normalize=normalize, missing_rate=missing_rate)
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
    window_size = train_config["window_size"]
    batch_size = train_config["batch_size"]
    val_split = train_config["val_split"]
    shuffle_dataset = train_config["shuffle_dataset"]
    train_dataset = SlidingWindowDataset(x_train, window_size, target_dims)
    test_dataset = SlidingWindowDataset(x_test, window_size, target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )
    return {"x_test": x_test, "x_train": x_train, "n_features": n_features, "train_loader": train_loader, "val_loader": val_loader, "test_loader": test_loader, "output_path": output_path, "target_dims": target_dims, "save_path": save_path, "y_test": y_test, "out_dim": out_dim}


def create_model(train_config, env_config, out_dim, n_features):

    model = GatGame(
        n_features=n_features,  # 特征数k
        window_size=train_config['window_size'],
        out_dim=out_dim,
        # kernel_size=args.kernel_size,
        use_gatv2=train_config['use_gatv2'],
        time_gat_embed_dim=train_config['time_gat_embed_dim'],
        dropout=train_config['dropout'],
        alpha=train_config['alpha'],
        device=env_config['device'],
        # mask_rate=args.mask_rate,
        mask_rate=train_config['mask_rate'],
        # replace_rate=args.replace_rate,
        replace_rate=train_config['replace_rate'],
        convEn=train_config['convEn'],
        remaskEn=train_config['remaskEn'],
        batch_size=train_config['batch_size'],
        topk=int(train_config['topk']),
        flow_hidden_size=train_config['flow_hidden_size'],
        flow_n_hidden=train_config['flow_n_hidden'],
        flow_n_blocks=train_config['flow_n_blocks'],
        rel=train_config['rel']
    )
    if env_config['device'] == "cuda:0":
        model.cuda()
    return model


def train(train_config, env_config, model, train_loader, val_loader, target_dims, output_path, id='0', n_features=1,model_dir='./Industrial_time_series_analysis/Diagnose/diagnose_utils/mgahgm_util/checkpoints'):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_config['init_lr'], weight_decay=train_config['weight_decay'], eps=train_config['eps'])
    print(model)
    log_dir = f'{output_path}/logs'
    trainer = Trainer(
        model,
        optimizer,
        train_config['window_size'],
        n_features,
        target_dims,
        train_config['epochs'],
        train_config['batch_size'],
        # init_lr,
        train_config['init_lr'],
        env_config['use_cuda'],
        model_dir,
        log_dir,
        train_config['print_every'],
        train_config['log_tensorboard'],
        loss_min_save=train_config['loss_min_save'],
        clip_grad=train_config['clip_grad'],
    )
    trainer.fit(train_loader, val_loader)
    return trainer
    # plot_losses(trainer.losses, save_path=save_path, plot=False)


def test(trainer, modelname, y_test, train_config, env_config, target_dims, x_train, x_test, n_features):
    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001),
        "SWAT": (0.99, 0.001),
        "WADI": (0.99, 0.001),
        "PSM": (0.9950, 0.001),
        "MSDS": (0.9950, 0.001)
    }
    key = "SMD-" + \
        train_config['group'][0] if env_config["dataset"] == "SMD" else env_config["dataset"]
    level, q = level_q_dict[key]
    if train_config['level'] is not None:
        level = train_config['level']
    if train_config['q'] is not None:
        q = train_config['q']

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1,
                      "SMD-3": 1, "WADI": 1, "SWAT": 1, "PSM": 1, "MSDS": 1}
    key = "SMD-" + \
        train_config['group'][0] if env_config["dataset"] == "SMD" else env_config["dataset"]
    reg_level = reg_level_dict[key]

    trainer.load(f"{modelname}.pt")
    prediction_args = {
        'dataset': env_config["dataset"],
        "target_dims": target_dims,
        'scale_scores': train_config['scale_scores'],
        "level": level,
        "q": q,
        'dynamic_pot': train_config['dynamic_pot'],
        "use_mov_av": train_config['use_mov_av'],
        "gamma": train_config['gamma'],
        "reg_level": reg_level,
        "save_path": modelname,
    }
    best_model = trainer.model
    window_size = train_config['window_size']
    predictor = Predictor(
        best_model,
        train_config['window_size'],
        n_features,
        prediction_args
    )

    label = y_test[window_size:] if y_test is not None else None
    e_eval, answer = predictor.predict_anomalies(x_train, x_test, label)
    return e_eval, answer


def main():
    # 设置参数
    dataset = "SMD"  # 假设数据集仍然是SWAT，如需更改，请替换此值
    group = "1-1"  # 假设组仍然是1-1，如需更改，请替换此值
    lookback = 100
    mask_rate = 0.45
    replace_rate = 0.25
    missing_rate = 0.0
    flow_hidden_size = 100
    flow_n_hidden = 1
    flow_n_blocks = 2
    epochs = 1
    init_lr = 0.01668
    weight_decay = 1e-5
    eps = 1e-4
    convEn = False
    remaskEn = True
    topk = 24
    loss_min_save = True
    clip_grad = True
    manual_seed = 42

    # 获取配置
    train_config, env_config = params_config(
        dataset=dataset,
        group=group,
        lookback=lookback,
        mask_rate=mask_rate,
        replace_rate=replace_rate,
        missing_rate=missing_rate,
        flow_hidden_size=flow_hidden_size,
        flow_n_hidden=flow_n_hidden,
        flow_n_blocks=flow_n_blocks,
        epochs=epochs,
        init_lr=init_lr,
        weight_decay=weight_decay,
        eps=eps,
        convEn=convEn,
        remaskEn=remaskEn,
        topk=topk,
        loss_min_save=loss_min_save,
        clip_grad=clip_grad,
        manual_seed=manual_seed
    )

    # 加载数据
    data_loaders = load_data(dataset, train_config)
    train_loader = data_loaders["train_loader"]
    val_loader = data_loaders["val_loader"]
    test_loader = data_loaders["test_loader"]
    output_path = data_loaders["output_path"]
    target_dims = data_loaders["target_dims"]
    save_path = data_loaders["save_path"]
    y_test = data_loaders["y_test"]
    out_dim = data_loaders["out_dim"]

    # 创建模型
    model = create_model(train_config=train_config, env_config=env_config,
                         out_dim=out_dim, n_features=data_loaders["n_features"])

    # 训练模型
    trainer = train(train_config=train_config, env_config=env_config, model=model, train_loader=train_loader,
                    val_loader=val_loader, target_dims=target_dims, output_path=output_path, n_features=data_loaders["n_features"])

    # 测试模型
    print(test(trainer=trainer, modelname=train_config["save_path"], y_test=y_test, train_config=train_config, env_config=env_config,
          target_dims=target_dims, x_train=data_loaders["x_train"], x_test=data_loaders["x_test"], n_features=data_loaders["n_features"]))


if __name__ == "__main__":
    main()
