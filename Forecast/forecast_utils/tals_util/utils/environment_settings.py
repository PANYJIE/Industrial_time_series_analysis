import os
import random
import math

import numpy as np
import tensorflow as tf


def tensorflow_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = '1'


def dataset_reshape(input_x, input_y):
    output_x, output_y = [], []
    assert len(input_x) == len(input_y)
    for i in range(len(input_x)):
        input_x_0 = input_x[i]
        input_y_0 = input_y[i]
        for j in range(input_x_0.shape[0]):
            output_x.append(input_x_0[j, :, :])
            output_y.append(input_y_0[j, :].reshape(-1, 1))
    return output_x, output_y


def mape(y_pred, y_true):
    if type(y_pred) is tf.Tensor:
        y_pred = y_pred.numpy()
    if type(y_true) is tf.Tensor:
        y_true = y_true.numpy()
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse(y_pred, y_true):
    if type(y_pred) is tf.Tensor:
        y_pred = y_pred.numpy()
    if type(y_true) is tf.Tensor:
        y_true = y_true.numpy()
    return math.sqrt(np.square(np.subtract(y_true, y_pred)).mean())


def mse(y_pred, y_true):
    if type(y_pred) is tf.Tensor:
        y_pred = y_pred.numpy()
    if type(y_true) is tf.Tensor:
        y_true = y_true.numpy()
    return np.mean((y_true - y_pred) ** 2)


def mae(y_pred, y_true):
    if type(y_pred) is tf.Tensor:
        y_pred = y_pred.numpy()
    if type(y_true) is tf.Tensor:
        y_true = y_true.numpy()
    return np.mean(np.abs(y_true - y_pred))
