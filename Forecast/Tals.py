import numpy as np
import pandas as pd
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from Industrial_time_series_analysis.Forecast.forecast_utils.tals_util.utils.LSTM import network_generation
from Industrial_time_series_analysis.Forecast.forecast_utils.tals_util.utils.environment_settings import tensorflow_seed, dataset_reshape, mape, rmse, mse, mae


def train(x_tr, y_tr, INPUT, OUTPUT):
    x_tr = np.array(x_tr)
    y_tr = np.array(y_tr)
    callback = EarlyStopping(patience=20)
    ta_lstm = network_generation(unit_type='lstm', attention=True, input_shape=INPUT, layers=3, units_per_layer=50,
                                 output_step=OUTPUT)
    ta_lstm.compile(Adam(learning_rate=0.005), loss='mse')
    ta_lstm.fit(x_tr, y_tr, batch_size=128, epochs=100, callbacks=[callback], validation_split=0.2)
    return ta_lstm


def test(ta_lstm, x_te, y_te, pred_normalizer: MinMaxScaler):
    x_te = np.array(x_te)
    y_te = np.array(y_te)
    y_te = np.squeeze(y_te)
    y_te_2 = pred_normalizer.inverse_transform(y_te)

    comparison_dict = dict(model_name=['TA_LSTM'], mse=[], rmse=[], mae=[], mape=[])
    y_pred = ta_lstm.predict(x_te)

    y_pred_inv_norm = pred_normalizer.inverse_transform(y_pred)

    MAE = mae(y_true=y_te_2, y_pred=y_pred_inv_norm)
    MAPE = mape(y_pred=y_pred_inv_norm, y_true=y_te_2)
    RMSE = rmse(y_pred=y_pred_inv_norm, y_true=y_te_2)
    MSE = mse(y_pred=y_pred_inv_norm, y_true=y_te_2)

    comparison_dict["mae"].append(MAE)
    comparison_dict["mape"].append(MAPE)
    comparison_dict["rmse"].append(RMSE)
    comparison_dict["mse"].append(MSE)
    return y_pred_inv_norm, comparison_dict

def sliding_window(processed_df: pd.DataFrame, input_len=15, output_len=3, stride=1, target="MLF"):
    x_window_set, y_window_set = [], []
    start_index = 0
    target_data = processed_df[target]
    for i in range(int((processed_df.shape[0] - input_len) / stride) - 1):
        x_start_index = start_index
        x_end_index = start_index + input_len

        y_start_index = x_end_index
        y_end_index = y_start_index + output_len

        if y_end_index > processed_df.shape[0]:
            y_end_index = processed_df.shape[0]
            y_start_index = y_end_index - output_len
            x_end_index = y_start_index
            x_start_index = x_end_index - input_len

        x_window = processed_df.iloc[x_start_index:x_end_index]
        x_window_set.append(np.array(x_window))

        y_window = target_data.iloc[y_start_index:y_end_index]
        y_window_set.append(np.array(y_window))

        start_index = start_index + stride
    x_window_set = np.array(x_window_set)
    y_window_set = np.array(y_window_set)
    return [x_window_set, y_window_set]

def data_segmentation(source: pd.DataFrame, target, input_len=15, output_len=3, stride=1, segment_col=None,
                      max_segments=None):
    x_list, y_list = [], []

    if segment_col:
        segment_list = source[segment_col].drop_duplicates()
        if max_segments:
            segment_list = segment_list[:max_segments]
        for segment in segment_list:
            segment_df = source[source[segment_col] == segment]
            segment_df = segment_df.drop(columns=[segment_col])
            x, y = sliding_window(processed_df=segment_df, input_len=input_len, output_len=output_len, stride=stride,
                                  target=target)
            x_list.append(x)
            y_list.append(y)

        x1, y1 = dataset_reshape(x_list, y_list)
    else:
        x, y = sliding_window(processed_df=source, input_len=input_len, output_len=output_len, stride=stride,
                              target=target)
        x_list.append(x)
        y_list.append(y)

        x1, y1 = dataset_reshape(x_list, y_list)

    return [x1, y1]


if __name__ == '__main__':
    tensorflow_seed(42)

    source_dataframe = pd.read_csv('../Data/S002_Processed.csv', index_col=0)
    source_dataframe=source_dataframe[source_dataframe["MLF"]>=20]
    print(source_dataframe.head())
    print(source_dataframe.info())

    minmax = MinMaxScaler()
    y_df = source_dataframe[["MLF"]]
    normalize_df = source_dataframe.drop(columns=["SLABNO"])
    normalize_df = minmax.fit_transform(normalize_df)
    source_dataframe.iloc[:, 1:] = normalize_df
    minmax2 = MinMaxScaler()
    data_test = minmax2.fit_transform(y_df)

    [x, y] = data_segmentation(source=source_dataframe, target='MLF', stride=3, segment_col='SLABNO', max_segments=2)

    normalizer: MinMaxScaler = minmax2

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print(x_test.shape)
    print(x_test)
    print(type(x_test))

    print(y_test.shape)
    print(type(y_test))
    print(y_test)

    ta_lstm = train(x_tr=x_train, y_tr=y_train, INPUT=[15,14], OUTPUT=3)
    y_pred_inv_norm, comparison_df = test(ta_lstm, x_te=x_test, y_te=y_test, pred_normalizer=normalizer)

    print(y_pred_inv_norm)
    print(comparison_df)
