import numpy as np
import pandas as pd

def train(df_data, predict_fea_index, model_name, total_epoch, window_length, horizon):

    from .forecast_utils.mcrn_util.functions import multi_to_one

    X = np.array(df_data.values, dtype=float)
    multi_to_one(X, predict_fea_index, model_name, total_epoch, window_length, horizon)

def test(test_df_data, model_name, window_length, horizon):
    from .forecast_utils.mcrn_util.functions import model_predict
    test_X = np.array(test_df_data.values, dtype=float)
    test_X = np.swapaxes(test_X, axis1=0, axis2=1)
    test_X = np.expand_dims(test_X, axis=0)
    test_preds, test_targets = model_predict(test_X, model_name, window_length, horizon)
    return test_preds, test_targets






