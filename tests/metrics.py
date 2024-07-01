import numpy as np

def rmse(x_err, y_err):
    return np.round(np.sqrt(np.mean((x_err - y_err) ** 2)), 4)

def mape(x_err, y_err):
    return np.round(np.mean(100 * np.abs(x_err - y_err) / y_err), 4)

def variance(x_err, y_err):
    return np.round(np.sum(y_err**2 - x_err), 4)
