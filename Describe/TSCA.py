# -*- coding:utf-8 -*-
"""
Name：PanYunJie
Date：2024-08-07
"""

from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc import GPDC
from tigramite import data_processing as pp
import pandas as pd
from statsmodels.tsa.api import VAR

def assess_stationary_and_diff_order(df, max_diff=5, significance_level=0.05):

    """
        Perform a stationarity test for each variable in the DataFrame, and if it is non-stationary,
        try to perform the difference until stationary, and give the difference order after stationarity.
    """

    stationary_results = {}

    for column in df.columns:
        time_series = df[column]
        diff_order = 0
        p_value = 1

        while diff_order <= max_diff and p_value > significance_level:
            if diff_order > 0:
                time_series = time_series.diff().dropna()
            adf_result = adfuller(time_series)
            p_value = adf_result[1]
            if p_value <= significance_level:
                stationary_results[column] = {'is_stationary': True, 'diff_order': diff_order}
                break
            diff_order += 1

        if p_value > significance_level:
            stationary_results[column] = {'is_stationary': False, 'diff_order': diff_order - 1}

    return stationary_results


def adf_test(selected_df):

    selected_df.set_index(['time'], inplace=True)
    # Normalization
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(selected_df)
    selected_df_scaled = pd.DataFrame(scaled_features, columns=selected_df.columns)

    stationary_results = assess_stationary_and_diff_order(selected_df_scaled)

    # Differential processing for smooth data
    df_diff_stable = pd.DataFrame(index=selected_df.index)
    for column, result in stationary_results.items():
        if result['is_stationary']:
            if result['diff_order'] == 0:
                df_diff_stable[column] = selected_df[column]
            else:
                df_diff_stable[column] = selected_df[column].diff(result['diff_order']).dropna()
        else:
            print(f"Variable '{column}' is still non-stationary after {result['diff_order']} order differencing.")

    # Remove any NaN values due to differential
    df_diff_stable.dropna(inplace=True)

    for variable, result in stationary_results.items():
        print(
            f"Variable '{variable}': Is Stationary? {result['is_stationary']}, Differencing Order Needed: {result['diff_order']}")
    df_diff_stable.reset_index(inplace=True)

    return df_diff_stable
def feature_selection(data, target_column, threshold):
    data = pd.DataFrame(data)
    print(data)


    # Check if the 'time' column is included
    if 'time' not in data.columns:
        raise ValueError("There is no 'time' column in the DataFrame")

    X = data.drop(columns=['time', target_column])
    y = data[target_column]

    # Computing Mutual Information (Regression Tasks)
    mi = mutual_info_regression(X, y)

    # Create a DataFrame that contains the feature name and mutual information values
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi})

    # Sort according to mutual information values in descending order
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

    # Select features with high mutual information values
    selected_features = mi_df[mi_df['Mutual Information'] > threshold]['Feature']
    selected_data = data[['time'] + selected_features.tolist() + [target_column]]

    return selected_features, selected_data
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体为Times New Roman
plt.rcParams.update({'font.size': 16})  # 设置全局字体大小
matplotlib.rcParams['axes.unicode_minus'] = False  # 对于负号的正确显示


def get_causality(data, tau_max, pc_alpha, alpha_level):

    data.index.name = 'time'

    # Convert Pandas DataFrame to the format required by Tigramite
    var_names = data.columns.tolist()
    data_new = data[var_names].values
    dataframe = pp.DataFrame(data_new, var_names=var_names)

    # Initialize the PCMCI and set up the conditional independence test
    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=GPDC(significance='analytic', gp_params=None),
                  verbosity=0)

    # Run the PCMCI algorithm
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha, alpha_level=alpha_level)
    """
        Note: The smaller the pc_alpha, the fewer parents you will get in the PC phase. 
        The smaller the alpha_level, the greater the likelihood of causality in the MCI phase.
    """

    # Extract salient links
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    sig_links = p_matrix <= alpha_level

    # Sets the edge width to represent the detected significance
    causal_effect = np.where(sig_links, val_matrix, 0)

    # # Draw a cause-and-effect diagram where the thickness of the edges represents the weights
    # tp.plot_graph(
    #     figsize=(6, 4),
    #     val_matrix=causal_effect,
    #     graph=sig_links,
    #     var_names=var_names,
    #     link_colorbar_label='Edge MCI',
    #     node_colorbar_label='Node MCI',
    #     link_width=causal_effect,  # Edge widths are adjusted by weight
    #     node_size=0.2,  # Set the node size
    # )
    # plt.show()
    #
    # for tau_index in range(1, tau_max + 1):
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(causal_effect[:, :, tau_index], annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1,
    #                 fmt=".3f", xticklabels=var_names, yticklabels=var_names)
    #     plt.xlabel(f'Effect variables')
    #     plt.ylabel('Cause variables')
    #     plt.title(f'Causal Effect Matrix at τ={tau_index}')
    #     plt.tight_layout()
    #     plt.show()

    # Store causal results in a DataFrame
    rows = []
    for i in range(causal_effect.shape[0]):
        for j in range(causal_effect.shape[1]):
            for tau_index in range(1, tau_max + 1):
                effect = causal_effect[i, j, tau_index]
                if effect != 0:
                    rows.append([var_names[i], var_names[j], tau_index, effect])

    causality_df = pd.DataFrame(rows, columns=['Cause', 'Effect', 'Lag', 'Causal Effect'])

    return causality_df

def lag_order(data):

    data.set_index(['time'], inplace=True)
    # 确保索引为 datetime 类型，并指定频率
    data.index = pd.to_datetime(data.index)
    data = data.asfreq('H')  # 假设您的数据频率为2小时，根据实际情况调整

    # 使用VAR模型和AIC准则确定最佳滞后阶数
    model = VAR(data)
    aic_values = []
    bic_values = []
    fpe_values = []
    hqic_values = []
    lags = range(1, 10)  # 可以根据需要调整最大滞后阶数的范围

    for lag in lags:
        result = model.fit(lag)
        aic_values.append(result.aic)
        bic_values.append(result.bic)
        fpe_values.append(result.fpe)
        hqic_values.append(result.hqic)

    # 找到AIC、BIC和HQIC最小的滞后阶数
    min_aic = min(aic_values)
    optimal_lag_aic = lags[aic_values.index(min_aic)]

    min_bic = min(bic_values)
    optimal_lag_bic = lags[bic_values.index(min_bic)]

    min_fpe = min(fpe_values)
    optimal_lag_fpe = lags[fpe_values.index(min_fpe)]

    min_hqic = min(hqic_values)
    optimal_lag_hqic = lags[hqic_values.index(min_hqic)]

    optimal_lag_list = [optimal_lag_aic, optimal_lag_bic, optimal_lag_fpe, optimal_lag_hqic]
    optimal_lag_list_new = list(set([x for x in optimal_lag_list if optimal_lag_list.count(x) > 1]))

    if not optimal_lag_list_new:
        optimal_lag_order = min(optimal_lag_list)
    elif len(optimal_lag_list_new) == 1:
        optimal_lag_order = optimal_lag_list_new[0]
    else:
        optimal_lag_order = min(optimal_lag_list_new)

    return optimal_lag_order
