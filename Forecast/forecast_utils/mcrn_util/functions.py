import numpy as np
from tsai.all import get_splits, TSClassification, TSStandardize, TSClassifier, accuracy, ClassificationInterpretation, load_learner
from tsai.all import TSRegressor, TSStandardize, SlidingWindow, MSELossFlat, rmse, load_learner, TSRegression, TimeSplitter,StandardScaler,mse
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import sys
import io
import pandas as pd

# 模型训练
def multi_to_one(X, predict_fea_index, model_name, total_epoch, window_length, horizon):
    # 数据分段
    indices_list = list(np.arange(X.shape[1]))
    y_list = [predict_fea_index]
    X, y = SlidingWindow(window_length, get_x=[item for item in indices_list if item not in y_list], get_y=predict_fea_index, horizon=horizon, stride=None)(X)
    # 数据集构建
    splits = get_splits(y, test_size=0.3, stratify=True, random_state=23, shuffle=True,  show_plot=False)
    tfms = [None, [TSRegression()]]
    batch_tfms = TSStandardize(by_sample=True)

    learn = TSRegressor(X, y, splits=splits, path='models', arch=model_name, tfms=tfms,
                        batch_tfms=batch_tfms, metrics=mse)
    # 训练模型
    learn.fit_one_cycle(total_epoch, 1e-3)
    # 保存数据
    learn.export(f'{model_name}.pkl')
    # learn.plot_metrics()


# 模型预测
def model_predict(X, model_name, window_length, horizon):
    learn = load_learner('./models/' + f'{model_name}.pkl')
    test_probas, test_targets, test_preds = learn.get_X_preds(X, with_decoded=True)
    test_preds = np.array(test_preds).astype(np.float32)
    # 保存预测文件
    # df = pd.DataFrame(test_preds)
    # df.to_excel(f'./output/predict.xlsx')
    return test_preds,test_targets



