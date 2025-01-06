import json
import time

import numpy as np
from tqdm import tqdm
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.eval_methods import *
from Industrial_time_series_analysis.Diagnose.diagnose_utils.mgahgm_util.utils import *


class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="summary.txt", score_weight=1):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name
        self.score_weight = score_weight

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """
        # values [28479, 38]
        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        flow_log_probs = []
        vae_log_probs = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)  # [256, 100, 38]
                y = y.to(device)  # [256, 1, 38]

                flow_log_prob, vae_log_prob = self.model(x)
                flow_log_probs.append(flow_log_prob.detach().cpu().numpy())
                vae_log_probs.append(vae_log_prob.detach().cpu().numpy())

        flow_log_probs = np.concatenate(flow_log_probs, axis=0)  # [28379, 38]
        vae_log_probs = np.concatenate(vae_log_probs, axis=0)  # [28379, 38]

        # np.save(f"output/flow_log_probs_{int(time.time())}", flow_log_probs)
        # np.save(f"output/vae_log_probs_{int(time.time())}", vae_log_probs)

        actual = values.detach().cpu().numpy()[
            self.window_size:]  # [28379, 38]

        # if self.target_dims is not None:
        #     actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)  # [28379, 38]
        df_dict = {}
        eps = 1e-14
        for i in range(flow_log_probs.shape[1]):
            # df_dict[f"Forecast_{i}"] = flow_log_probs[:, i]
            # df_dict[f"Recon_{i}"] = vae_log_probs[:, i]
            df_dict[f"flow_log_probs_{i}"] = flow_log_probs[:, i]
            df_dict[f"vae_log_probs_{i}"] = vae_log_probs[:, i]
            df_dict[f"True_{i}"] = actual[:, i]
            flow_log_probs[:, i] = np.exp(flow_log_probs[:, i])
            flow_log_probs[:, i] = 1-(flow_log_probs[:, i] - np.min(flow_log_probs[:, i])) / (
                np.max(flow_log_probs[:, i])-np.min(flow_log_probs[:, i])+eps)
            vae_log_probs[:, i] = np.exp(vae_log_probs[:, i])
            vae_log_probs[:, i] = 1-(vae_log_probs[:, i] - np.min(vae_log_probs[:, i])) / (
                np.max(vae_log_probs[:, i]) - np.min(vae_log_probs[:, i]) + eps)
            a_score = (flow_log_probs[:, i] + vae_log_probs[:, i])
            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1+iqr)

            anomaly_scores[:, i] = a_score
            df_dict[f"A_Score_{i}"] = a_score

        df = pd.DataFrame(df_dict)
        anomaly_scores = np.mean(anomaly_scores, 1)

        df['A_Score_Global'] = anomaly_scores

        return df

    def predict_anomalies(self, train, test, true_anomalies, load_scores=False, save_output=True,
                          scale_scores=False):
        """ Predicts anomalies

        :param train: 2D array of train multivariate time series data
        :param test: 2D array of test multivariate time series data
        :param true_anomalies: true anomalies of test set, None if not available
        :param save_scores: Whether to save anomaly scores of train and test
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        :param scale_scores: Whether to feature-wise scale anomaly scores
        """

        if load_scores:
            print("Loading anomaly scores")

            train_pred_df = pd.read_pickle(
                f"{self.save_path}/train_output.pkl")
            test_pred_df = pd.read_pickle(f"{self.save_path}/test_output.pkl")

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

        else:
            train_pred_df = self.get_score(train)
            test_pred_df = self.get_score(test)

            train_anomaly_scores = train_pred_df['A_Score_Global'].values
            test_anomaly_scores = test_pred_df['A_Score_Global'].values

            train_anomaly_scores = adjust_anomaly_scores(
                train_anomaly_scores, self.dataset, True, self.window_size)
            test_anomaly_scores = adjust_anomaly_scores(
                test_anomaly_scores, self.dataset, False, self.window_size)

            # Update df
            train_pred_df['A_Score_Global'] = train_anomaly_scores
            test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            # 指数加权移动平均
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            train_anomaly_scores = pd.DataFrame(train_anomaly_scores).ewm(
                span=smoothing_window).mean().values.flatten()
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(
                span=smoothing_window).mean().values.flatten()

        # Find threshold and predict anomalies at feature-level (for plotting and diagnosis purposes)
        out_dim = self.n_features if self.target_dims is None else len(
            self.target_dims)
        all_preds = np.zeros((len(test_pred_df), out_dim))
        for i in range(out_dim):
            train_feature_anom_scores = train_pred_df[f"A_Score_{i}"].values
            test_feature_anom_scores = test_pred_df[f"A_Score_{i}"].values
            epsilon = find_epsilon(train_feature_anom_scores, reg_level=2)

            train_feature_anom_preds = (
                train_feature_anom_scores >= epsilon).astype(int)
            test_feature_anom_preds = (
                test_feature_anom_scores >= epsilon).astype(int)

            train_pred_df[f"A_Pred_{i}"] = train_feature_anom_preds
            test_pred_df[f"A_Pred_{i}"] = test_feature_anom_preds

            train_pred_df[f"Thresh_{i}"] = epsilon
            test_pred_df[f"Thresh_{i}"] = epsilon

            all_preds[:, i] = test_feature_anom_preds

        # Global anomalies (entity-level) are predicted using aggregation of anomaly scores across all features
        # These predictions are used to evaluate performance, as true anomalies are labeled at entity-level
        # Evaluate using different threshold methods: brute-force, epsilon and peaks-over-treshold
        # e_eval = epsilon_eval(train_anomaly_scores, test_anomaly_scores,
        #                       true_anomalies, reg_level=self.reg_level)
        # p_eval = pot_eval(train_anomaly_scores, test_anomaly_scores, true_anomalies,
        #                   q=self.q, level=self.level, dynamic=self.dynamic_pot)
        if true_anomalies is not None:
            # bf_eval = bf_search(test_anomaly_scores, true_anomalies, start=0.01, end=2, step_num=100, verbose=False)
            bf_eval = bf_search(test_anomaly_scores, true_anomalies,
                                start=0, end=2, step_num=2000, verbose=False)
        else:
            bf_eval = {}

        # print(f"Results using epsilon method:\n {e_eval}")
        # print(f"Results using peak-over-threshold method:\n {p_eval}")
        print(f"Results using best f1 score search:\n {bf_eval}")

        # for k, v in e_eval.items():
        #     if not type(e_eval[k]) == list:
        #         e_eval[k] = float(v)
        # for k, v in p_eval.items():
        #     if not type(p_eval[k]) == list:
        #         p_eval[k] = float(v)
        for k, v in bf_eval.items():
            bf_eval[k] = float(v)

        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
        if save_output:
            global_epsilon = bf_eval["threshold"]
            test_pred_df["A_True_Global"] = true_anomalies
            train_pred_df["Thresh_Global"] = global_epsilon
            test_pred_df["Thresh_Global"] = global_epsilon
            train_pred_df[f"A_Pred_Global"] = (
                train_anomaly_scores >= global_epsilon).astype(int)
            test_preds_global = (test_anomaly_scores >=
                                 global_epsilon).astype(int)
            # Adjust predictions according to evaluation strategy
            if true_anomalies is not None:
                test_preds_global = adjust_predicts(
                    None, true_anomalies, global_epsilon, pred=test_preds_global)
            test_pred_df[f"A_Pred_Global"] = test_preds_global

            # print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
            # train_pred_df.to_pickle(f"{self.save_path}/train_output.pkl")
            # test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")
        return bf_eval, (test_anomaly_scores >= global_epsilon).astype(int)
        print("-- Done.")
