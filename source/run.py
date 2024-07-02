from functools import partial
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from tests.metrics import mape, rmse, variance
from tests.test import test
from .simulation import random_linear_func, random_gaussian_mixture_func, random_GMM_samples, random_uniform_samples

class DummyModel:
    def __init__(self):
        self.func = None

    def fit(self, X, y):
        self.func = random_linear_func({'n_dim': X.shape[1], 'max_mu': np.max(X.sum(axis=1))})

    def predict(self, X):
        return self.func(X)
    
def run_test_case(conf, f_gen, model, g_gen, p_gen, n_tests, n_splits, x, y,
                  save_fig = False,
                  show_fig = False,
                  return_metrics = False,
                  hyperparams = {'kde_size' : ['scott'],
                                 'epsilon_reg' : [0],
                                 'epsilon_clip' : [0]}):
    
    log_err = test(conf, f_gen=f_gen, model = model, g_gen = g_gen, p_gen = p_gen, n_tests = n_tests, n_splits = n_splits,
                   target_error = x + [y], hyperparams = hyperparams)
    
    metrics_dict = {'mape' : {},
                   'rmse' : {},
                   'variance' : {}}
    
    y_err = np.exp(log_err[y])
    for x_temp in x:
        x_err = np.exp(log_err[x_temp])
        metrics_dict['mape'][x_temp] = mape(x_err, y_err)
        metrics_dict['rmse'][x_temp] = rmse(x_err, y_err)
        metrics_dict['variance'][x_temp] = variance(x_err, y_err)

    if return_metrics:
        return metrics_dict

def run(conf, f, model, n_splits, x, y, n_tests = 1,
        hyperparams = {'kde_size' : ['scott'],
                        'epsilon_reg' : [0],
                        'epsilon_clip' : [0]}):
    
    f_gens = {'linear': partial(random_linear_func, conf),
              'GMM': partial(random_gaussian_mixture_func, conf)}

    models = {'linear': DummyModel(),
              'boosting': GradientBoostingRegressor()}

    return run_test_case(conf, f_gen = f_gens[f],
                  model = models[model],
                  g_gen = partial(random_GMM_samples, conf),
                  p_gen = partial(random_uniform_samples, conf, True),
                  n_tests = n_tests,
                  n_splits = n_splits,
                  x = x,
                  y = y,
                  show_fig = True,
                  save_fig = False,
                  return_metrics = True,
                  hyperparams = hyperparams)
