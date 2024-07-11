from functools import partial
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from tqdm import trange
import logging

from tests.metrics import mape, rmse, variance
from tests.test import test
from .simulation import (
    random_linear_func,
    random_gaussian_mixture_func,
    random_GMM_samples,
    random_uniform_samples,
    DummyModel,
)

def run(
    conf,
    params,
    hyperparams_params,
    hyperparams_dict={
        "kde_size": [5],
        "n_slices": [3],
        "ISE_g_regular": [0],
        "ISE_g_clip": [0],
        "ISE_g_estim_clip": [0],
    },
):
    f_gens = {
        "linear": partial(random_linear_func, conf),
        "GMM": partial(random_gaussian_mixture_func, conf),
    }

    models = {"linear": DummyModel(), "boosting": GradientBoostingRegressor()}

    params['f_gen'] = f_gens[params['f']]
    params['model_gen'] = models[params['model']]
    params['g_gen'] = partial(random_GMM_samples, conf)
    params['p_gen'] = partial(random_uniform_samples, conf, True)
            
    best_hyperparams = {
        "kde_size": hyperparams_dict["kde_size"][0],
        "n_slices": hyperparams_dict["Mandoline"][0],
        "ISE_g_regular": hyperparams_dict["ISE_g_regular"][0],
        "ISE_g_clip": hyperparams_dict["ISE_g_clip"][0],
        "ISE_g_estim_clip": hyperparams_dict["ISE_g_clip"][0],
    }

    if hyperparams_params['grid_flag']:
        x_hyp = ["ISE_g_regular", "ISE_g_estim_clip", "ISE_g_clip", "Mandoline"]
        metrics_list_hyp, sizes = hyperparams_search(
            conf,
            params,
            hyperparams_params,
            hyperparams_dict,
            x_hyp
        )
        best_hyperparams, mape_dict, hyp_dict = find_best_hyp(
            x_hyp, metrics_list_hyp, sizes
        )
        extr_plots(conf, x_hyp, mape_dict, hyp_dict)

        logging.info(f"for max_cov={conf['max_cov']}:")
        logging.info(f"mape_dict:\n {mape_dict}")
        logging.info(f"hyperparam dict:\n {hyp_dict}")
        logging.info(f"best_hyperparams:\n{best_hyperparams}")
    else:
        logging.info(
            f"Running with default without GridSearch hyperparams:\n{best_hyperparams}"
        )

    logging.info("TEST FOR PLOT WITH BEST HYPERPARAMS:\n")
    log_err = test(
        conf,
        params,
        hyperparams_params,
        target_error=params['xs'] + params['y'],
        hyperparams=best_hyperparams,
    )

    metrics_dict = {"mape": {}, "rmse": {}}
    y_err = np.exp(log_err[params['y'][0]])
    for x_temp in params['xs']:
        x_err = np.exp(log_err[x_temp])
        metrics_dict["mape"][x_temp] = mape(x_err, y_err)
        metrics_dict["rmse"][x_temp] = rmse(x_err, y_err)
    return metrics_dict

def hyperparams_search(
   conf, params, hyperparams_params, hyperparams_dict, x_hyp):
    params['n_tests'] = hyperparams_params['n_hyp_tests']
    hyperparams = {"kde_size": hyperparams_dict["kde_size"][0]}
    metrics_list_hyp = []

    sizes = dict()
    max_size = 0
    for x_temp in x_hyp:
        sizes[x_temp] = len(hyperparams_dict[x_temp])
        if len(hyperparams_dict[x_temp]) > max_size:
            max_size = len(hyperparams_dict[x_temp])

    print("TEST FOR HYPERPARAMS SEARCH:")
    log_err_hyp_list = []
    for i in trange(max_size):
        hyperparams = {"kde_size": hyperparams_dict["kde_size"][0]}
        x_hyp_temp = []
        for x_temp in x_hyp:
            if i < sizes[x_temp]:
                hyperparams[x_temp] = hyperparams_dict[x_temp][i]
                x_hyp_temp += [x_temp]

        log_err_hyp_list += [
            test(
                conf,
                params,
                hyperparams_params,
                target_error=x_hyp_temp + params['y'],
                hyperparams=hyperparams,
            )
        ]

        metrics_dict = {"mape": {}, "rmse": {}}

        y_err = np.exp(log_err_hyp_list[i][params['y'][0]])
        for x_temp in x_hyp_temp:
            x_err = np.exp(log_err_hyp_list[i][x_temp])
            metrics_dict["mape"][x_temp] = mape(x_err, y_err)
            metrics_dict["rmse"][x_temp] = rmse(x_err, y_err)

        metrics_list_hyp += [(hyperparams, metrics_dict)]

    return metrics_list_hyp, sizes

def extr_plots(conf, x_hyp, mape_dict, hyp_dict):
    for x_temp in x_hyp:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(hyp_dict[x_temp], mape_dict[x_temp], label=f"{x_temp}")
        plt.legend(fontsize=26)
        plt.title(f"max_cov{conf['max_cov']}")
        ax.set_xlabel(f"param for {x_temp}", fontsize=26)
        ax.set_ylabel("mape", fontsize=26)
        # ax.set_xscale('log')
        plt.savefig(f"./plots/results/{x_temp}_{conf['max_cov']}.pdf")
        plt.tight_layout()
        # plt.show()

def find_best_hyp(x_hyp, metrics_list_hyp, sizes):
    best_hyperparams = {"kde_size": metrics_list_hyp[0][0]["kde_size"]}
    hyp_dict = {}
    mape_dict = {}
    for x_temp in x_hyp:
        hyp_dict[x_temp] = []
        mape_dict[x_temp] = []

    print(metrics_list_hyp)
    for x_temp in x_hyp:
        error = 0
        for i in range(sizes[x_temp]):
            hyp_dict[x_temp] += [metrics_list_hyp[i][0][x_temp]]

            error = metrics_list_hyp[i][1]["mape"][x_temp]
            if np.isnan(error):
                error = np.inf
            mape_dict[x_temp] += [error]

        best_hyperparams[x_temp] = hyp_dict[x_temp][
            mape_dict[x_temp].index(min(mape_dict[x_temp]))
        ]

    return best_hyperparams, mape_dict, hyp_dict

