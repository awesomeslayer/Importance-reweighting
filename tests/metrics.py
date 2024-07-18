import numpy as np


def rmse(x_err, y_err):
    return np.sqrt(np.mean((x_err - y_err) ** 2))


def mape(x_err, y_err):
    return np.mean(100 * np.abs(x_err - y_err) / y_err)
