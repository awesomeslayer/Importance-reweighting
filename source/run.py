from functools import partial
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
from tqdm import trange
import logging
import hydra

from tests.metrics import mape, rmse, variance
from tests.test import test
from .simulation import (
    random_linear_func,
    random_gaussian_mixture_func,
    random_GMM_samples,
    random_uniform_samples,
    DummyModel,
)


def find_best_hyp(x_hyp, metrics_list_hyp):
    best_hyperparams = {"kde_size": metrics_list_hyp[0][0]["kde_size"]}
    hyp_dict = {}
    mape_dict = {}
    for x_temp in x_hyp:
        hyp_dict[x_temp] = []
        mape_dict[x_temp] = []

    for x_temp in x_hyp:
        error = 0
        for i in range(len(metrics_list_hyp)):
            hyp_dict[x_temp] += [metrics_list_hyp[i][0][x_temp]]

            error = metrics_list_hyp[i][1]["mape"][x_temp]
            if np.isnan(error):
                error = np.inf
            mape_dict[x_temp] += [error]

        best_hyperparams[x_temp] = hyp_dict[x_temp][
            mape_dict[x_temp].index(min(mape_dict[x_temp]))
        ]

    return best_hyperparams, mape_dict, hyp_dict


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


def hyperparams_search(
    conf, f_gen, model, g_gen, p_gen, n_hyp_tests, n_splits, x_hyp, y, hyperparams_dict
):

    hyperparams = {"kde_size": hyperparams_dict["kde_size"][0]}
    metrics_list_hyp = []
    size = len(hyperparams_dict["ISE_g_regular"])  #
    for x_temp in x_hyp:
        if size != len(hyperparams_dict[x_temp]):
            print("ERROR, LENGTHS OF PARAMS MUST BE SIMILAR")
            return "ERROR"
    log_err_hyp_list = []

    print("TEST FOR HYPERPARAMS SEARCH:")
    for i in trange(size):

        hyperparams = {"kde_size": hyperparams_dict["kde_size"][0]}
        for x_temp in x_hyp:
            hyperparams[x_temp] = hyperparams_dict[x_temp][i]

        log_err_hyp_list += [
            test(
                conf,
                f_gen=f_gen,
                model=model,
                g_gen=g_gen,
                p_gen=p_gen,
                n_tests=n_hyp_tests,
                n_splits=n_splits,
                target_error=x_hyp + [y],
                hyperparams=hyperparams,
            )
        ]

        metrics_dict = {"mape": {}, "rmse": {}}
        y_err = np.exp(log_err_hyp_list[i][y])
        for x_temp in x_hyp:
            x_err = np.exp(log_err_hyp_list[i][x_temp])
            metrics_dict["mape"][x_temp] = mape(x_err, y_err)
            metrics_dict["rmse"][x_temp] = rmse(x_err, y_err)

        metrics_list_hyp += [(hyperparams, metrics_dict)]

    return metrics_list_hyp, log_err_hyp_list


def run_test_case(
    conf,
    f_gen,
    model,
    g_gen,
    p_gen,
    n_tests,
    n_splits,
    x,
    y,
    n_hyp_tests=5,
    hyperparams_dict={
        "kde_size": [5],
        "ISE_g_regular": [0],
        "ISE_g_clip": [0],
        "ISE_g_estim_clip": [0],
    },
    grid_flag=True,
):
    best_hyperparams = {
        "kde_size": hyperparams_dict["kde_size"][0],
        "ISE_g_regular": hyperparams_dict["ISE_g_regular"][0],
        "ISE_g_clip": hyperparams_dict["ISE_g_clip"][0],
        "ISE_g_estim_clip": hyperparams_dict["ISE_g_clip"][0],
    }
    if grid_flag:
        x_hyp = ["ISE_g_regular", "ISE_g_estim_clip", "ISE_g_clip"]
        metrics_list_hyp, log_err_hyp_list = hyperparams_search(
            conf,
            f_gen,
            model,
            g_gen,
            p_gen,
            n_hyp_tests,
            n_splits,
            x_hyp,
            y,
            hyperparams_dict,
        )
        best_hyperparams, mape_dict, hyp_dict = find_best_hyp(x_hyp, metrics_list_hyp)
        extr_plots(conf, x_hyp, mape_dict, hyp_dict)

        logging.debug(f"for max_cov={conf['max_cov']}:")
        logging.debug(f"mape_dict:\n {mape_dict}")
        logging.debug(f"hyperparam dict:\n {hyp_dict}")
        logging.debug(f"best_hyperparams:\n{best_hyperparams}")
    else:
        logging.debug(
            f"Running with default without GridSearch hyperparams:\n{best_hyperparams}"
        )

    logging.info("TEST FOR PLOT WITH BEST HYPERPARAMS:\n")
    log_err = test(
        conf,
        f_gen=f_gen,
        model=model,
        g_gen=g_gen,
        p_gen=p_gen,
        n_tests=n_tests,
        n_splits=n_splits,
        target_error=x + [y],
        hyperparams=best_hyperparams,
    )

    metrics_dict = {"mape": {}, "rmse": {}}
    y_err = np.exp(log_err[y])
    for x_temp in x:
        x_err = np.exp(log_err[x_temp])
        metrics_dict["mape"][x_temp] = mape(x_err, y_err)
        metrics_dict["rmse"][x_temp] = rmse(x_err, y_err)
    return metrics_dict


def run(
    conf,
    f,
    model,
    n_splits,
    x,
    y,
    n_tests=5,
    n_hyp_tests=5,
    hyperparams_dict={
        "kde_size": [5],
        "ISE_g_regular": [0],
        "ISE_g_clip": [0],
        "ISE_g_estim_clip": [0],
    },
    grid_flag=True,
):

    f_gens = {
        "linear": partial(random_linear_func, conf),
        "GMM": partial(random_gaussian_mixture_func, conf),
    }

    models = {"linear": DummyModel(), "boosting": GradientBoostingRegressor()}

    return run_test_case(
        conf,
        f_gen=f_gens[f],
        model=models[model],
        g_gen=partial(random_GMM_samples, conf),
        p_gen=partial(random_uniform_samples, conf, True),
        n_tests=n_tests,
        n_splits=n_splits,
        x=x,
        y=y,
        n_hyp_tests=n_hyp_tests,
        hyperparams_dict=hyperparams_dict,
        grid_flag=grid_flag,
    )
