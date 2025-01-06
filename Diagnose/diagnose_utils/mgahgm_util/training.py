import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    """Trainer class for MTAD-GAT model.

    :param model: MTAD-GAT model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param forecast_criterion: Loss to be used for forecasting.
    :param recon_criterion: Loss to be used for reconstruction.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=200,
        batch_size=256,
        init_lr=0.001,
        use_cuda=True,
        save_path="",
        log_dir="output/",
        print_every=1,
        log_tensorboard=True,
        loss_min_save=False,
        clip_grad=True,

    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.save_path = save_path
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard
        self.loss_min_save = loss_min_save
        self.clip_grad = clip_grad
        self.save_name = "model.pt"
        #     "train_total": [],
        #     "train_forecast": [],
        #     "train_recon": [],
        #     "val_total": [],
        #     "val_forecast": [],
        #     "val_recon": [],
        # }
        self.losses = {
            "train_total": [],
            "train_flow": [],
            "train_vae": [],
            "val_total": [],
            "val_flow": [],
            "val_vae": [],
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

            # self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader, val_loader=None):
        """Train model for self.n_epochs.
        Train and validation (if validation loader given) losses stored in self.losses

        :param train_loader: train loader of input data
        :param val_loader: validation loader of input data
        """
        init_train_loss = self.evaluate(train_loader)
        print(f"Init total train loss: {init_train_loss[2]:5f}")

        if val_loader is not None:
            init_val_loss = self.evaluate(val_loader)
            print(f"Init total val loss: {init_val_loss[2]:.5f}")

        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()

        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            self.model.train()
            flow_losses = []
            vae_losses = []
            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)

                if y.ndim == 3:
                    y = y.squeeze(1)

                loss = self.model.total_loss
                loss.backward()
                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()

                flow_losses.append(self.model.flow_loss.item())
                vae_losses.append(self.model.vae_loss.item())
            flow_losses = np.array(flow_losses)
            vae_losses = np.array(vae_losses)
            flow_epoch_loss = flow_losses.mean()
            vae_epoch_loss = vae_losses.mean()
            total_epoch_loss = flow_epoch_loss + vae_epoch_loss

            self.losses["train_flow"].append(flow_epoch_loss)
            self.losses["train_vae"].append(vae_epoch_loss)
            self.losses["train_total"].append(total_epoch_loss)

            # Evaluate on validation set
            flow_val_loss, vae_val_loss, total_val_loss = "NA", "NA", "NA"
            if val_loader is not None:
                flow_val_loss, vae_val_loss, total_val_loss = self.evaluate(
                    val_loader)
                self.losses["val_flow"].append(flow_val_loss)
                self.losses["val_vae"].append(vae_val_loss)
                self.losses["val_total"].append(total_val_loss)

                if self.loss_min_save:
                    if total_val_loss <= min(self.losses["val_total"]):
                        self.save(self.save_name)

                elif total_val_loss <= self.losses["val_total"][-1]:
                    self.save(self.save_name)
            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
#             '''
# ==============================nni====================================
#             # '''
#             # nni.report_intermediate_result(total_val_loss)
#             # '''
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^nni^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#             '''
            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"flow_loss = {flow_epoch_loss:.5f}, "
                    f"vae_loss = {vae_epoch_loss:.5f}, "
                    f"total_loss = {total_epoch_loss:.5f}"
                )
                if val_loader is not None:
                    s += (
                        f" ---- val_flow_loss = {flow_val_loss:.5f}, "
                        f"val_vae_loss = {vae_val_loss:.5f}, "
                        f"val_total_loss = {total_val_loss:.5f}"
                    )
                s += f" [{epoch_time:.1f}s]"
                print(s)

        if val_loader is None:
            self.save(self.save_path+'/model.pt')
        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            print("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")

    def evaluate(self, data_loader):
        """Evaluate model

        :param data_loader: data loader of input data
        :return flow_loss, vae_loss, total_loss
        """
        self.model.eval()

        flow_losses = []
        vae_losses = []
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)

                if self.target_dims is not None:
                    x = x[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                if y.ndim == 3:
                    y = y.squeeze(1)
                flow_losses.append(self.model.flow_loss.item())
                vae_losses.append(self.model.vae_loss.item())

        flow_losses = np.array(flow_losses)
        vae_losses = np.array(vae_losses)
        flow_loss = flow_losses.mean()
        vae_loss = vae_losses.mean()

        total_loss = flow_loss + vae_loss
        return flow_loss, vae_loss, total_loss

    # def save(self, file_name):
    #     """
    #     Pickles the model parameters to be retrieved later
    #     :param file_name: the filename to be saved as,`dload` serves as the download directory
    #     """
    #     PATH =file_name
    #     torch.save(self.model.state_dict(), PATH)

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        # 获取当前脚本所在目录的路径
        base_dir = os.path.dirname(__file__)
        # 构建保存路径，将文件保存在与training.py同级目录下的checkpoints目录中
        save_dir = os.path.join(base_dir, 'checkpoints')
        # 如果目录不存在，则创建它
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 最终的保存路径
        save_path = os.path.join(save_dir, file_name)
        # 保存模型参数
        torch.save(self.model.state_dict(), save_path)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                print(key, value[-1], epoch)
