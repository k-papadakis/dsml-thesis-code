import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def smape(y_true, y_pred):
    return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def mdape(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true))


def mase(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / np.mean(np.abs(y_true[1:] - y_true[:-1])))


METRICS = (mse, rmse, mae, mape, mdape, smape, mase)
