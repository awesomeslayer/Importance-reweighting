from functools import partial
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import logging

from tests.metrics import mape, rmse, variance
from tests.test import test
from .simulation import (
    random_linear_func,
    random_gaussian_mixture_func,
    random_GMM_samples,
    random_uniform_samples,
    random_GP_func,
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
        "GP": partial(random_GP_func, conf),
    }

    models = {"linear": DummyModel(), "boosting": GradientBoostingRegressor()}

    params["f_gen"] = f_gens[params["f"]]
    params["model_gen"] = models[params["model"]]
    params["g_gen"] = partial(random_GMM_samples, conf)
    params["p_gen"] = partial(random_uniform_samples, conf, True)

    best_hyperparams = {
        "kde_size": hyperparams_dict["kde_size"][0],
        "n_slices": hyperparams_dict["Mandoline"][0],
        "ISE_g_regular": hyperparams_dict["ISE_g_regular"][0],
        "ISE_g_clip": hyperparams_dict["ISE_g_clip"][0],
        "ISE_g_estim_clip": hyperparams_dict["ISE_g_clip"][0],
    }

    kf = (
        KFold(n_splits=params["n_splits"])
        if params["n_splits"] > 1
        else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    )

    sizes = dict()
    max_size = 0
    for x_temp in params["x_hyp"]:
        sizes[x_temp] = len(hyperparams_dict[x_temp])
        if len(hyperparams_dict[x_temp]) > max_size:
            max_size = len(hyperparams_dict[x_temp])
    logging.info(f"max_size = {max_size}\n sizes = {sizes}")

    gen_params = {}

    log_err_dict = {}
    for x_temp in params["y"] + params["x"]:
        log_err_dict[x_temp] = []

    logging.info(f"log_err_dict_init: {log_err_dict}")

    log_err_hyp_dict = {}
    for x_temp in params["x_hyp"]:
        log_err_hyp_dict[x_temp] = []
        for i in range(sizes[x_temp]):
            log_err_hyp_dict[x_temp].append([])

    logging.info(f"log_err_hyp_dict init: {log_err_hyp_dict}")

    metrics_dict = {"mape": {}, "rmse": {}}
    for i in trange(params["n_tests"]):
        # gen for 1 test
        gen_params["f"] = params["f_gen"]()
        gen_params["g_sample"], gen_params["g"] = params["g_gen"]()
        gen_params["p_sample"], gen_params["p"] = params["p_gen"]()

        # work without hyperparams:
        for x_temp in params["y"] + params["x"]:
            log_err_dict[x_temp] += [
                test(
                    conf,
                    params,
                    hyperparams_params,
                    gen_params,
                    kf,
                    params["y"] + params["x"],
                    best_hyperparams,
                )[x_temp]
            ]

        # logging.info(f"n_test:{i} current log-errors = {log_err_dict}")
        y_err = np.exp(log_err_dict[params["y"][0]])

        for x_temp in params["x"]:
            x_err = np.exp(log_err_dict[x_temp])
            metrics_dict["mape"][x_temp] = mape(x_err, y_err)
            metrics_dict["rmse"][x_temp] = rmse(x_err, y_err)

        # hyperparams:
        if hyperparams_params["grid_flag"]:
            for i in trange(max_size):
                hyperparams = {"kde_size": hyperparams_dict["kde_size"][0]}
                x_estim_temp = []
                for x_temp in params["x_hyp"]:
                    if i < sizes[x_temp]:
                        hyperparams[x_temp] = hyperparams_dict[x_temp][i]
                        x_estim_temp += [x_temp]

                for x_temp in x_estim_temp:
                    log_err_hyp_dict[x_temp][i] += [
                        test(
                            conf,
                            params,
                            hyperparams_params,
                            gen_params,
                            kf,
                            x_estim_temp,
                            hyperparams,
                        )[x_temp]
                    ]

            # logging.info(f"ntest{i}: current log_errors_hyp_dict: {log_err_hyp_dict}")
        else:  # without search
            logging.info("grid flag is off, no hyp-search:")
            for x_temp in params["x_hyp"]:
                log_err_dict[x_temp] += [
                    [
                        test(
                            conf,
                            params,
                            hyperparams_params,
                            gen_params,
                            kf,
                            params["x_hyp"],
                            best_hyperparams,
                        )
                    ][x_temp]
                ]

    logging.info(f"final log_err_dict: {log_err_dict}")
    logging.info(f"final log_err_hyp_dict: {log_err_hyp_dict}")

    # resulting
    if hyperparams_params["grid_flag"]:
        best_metrics_estim_dict, metrics_estim_dict = find_best_params(
            params, hyperparams_dict, sizes, log_err_hyp_dict, log_err_dict
        )
        extr_plots(conf, params, metrics_estim_dict, hyperparams_dict)

    for x_temp in params["x_hyp"]:
        metrics_dict["mape"][x_temp] = best_metrics_estim_dict["mape"][x_temp]
        metrics_dict["rmse"][x_temp] = best_metrics_estim_dict["rmse"][x_temp]
    logging.info(f"final_metrics_dict: {metrics_dict}")
    return metrics_dict


def find_best_params(params, hyperparams_dict, sizes, log_err_hyp_dict, log_err_dict):
    best_hyperparams = {}
    best_metrics_dict = {"mape": {}, "rmse": {}}
    metrics_hyp_dict = {"mape": {}, "rmse": {}}

    for x_temp in params["x_hyp"]:
        metrics_hyp_dict["mape"][x_temp] = []
        metrics_hyp_dict["rmse"][x_temp] = []

    y_err = np.exp(log_err_dict[params["y"][0]])
    for x_temp in params["x_hyp"]:
        for i in range(sizes[x_temp]):
            x_err = np.exp(log_err_hyp_dict[x_temp][i])
            metrics_hyp_dict["mape"][x_temp] += [mape(x_err, y_err)]
            metrics_hyp_dict["rmse"][x_temp] += [rmse(x_err, y_err)]

        best_hyperparams[x_temp] = hyperparams_dict[x_temp][
            metrics_hyp_dict["mape"][x_temp].index(
                min(metrics_hyp_dict["mape"][x_temp])
            )
        ]
        best_metrics_dict["mape"][x_temp] = metrics_hyp_dict["mape"][x_temp][
            metrics_hyp_dict["mape"][x_temp].index(
                min(metrics_hyp_dict["mape"][x_temp])
            )
        ]
        best_metrics_dict["rmse"][x_temp] = metrics_hyp_dict["rmse"][x_temp][
            metrics_hyp_dict["mape"][x_temp].index(
                min(metrics_hyp_dict["mape"][x_temp])
            )
        ]

    logging.info(f"metrics_dict_hyp: {metrics_hyp_dict}")
    logging.info(f"best hyperparams: {best_hyperparams}")
    logging.info(f"best metrics dict: {best_metrics_dict}")
    return best_metrics_dict, metrics_hyp_dict


def extr_plots(conf, params, metrics_hyp_dict, hyperparams_dict):
    for x_temp in params["x_hyp"]:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(
            hyperparams_dict[x_temp],
            metrics_hyp_dict["mape"][x_temp],
            label=f"{x_temp}",
        )
        plt.legend(fontsize=26)
        plt.title(f"gens_{params['model']}_{params['f']}_max_cov{conf['max_cov']}")
        ax.set_xlabel(f"param for {x_temp}", fontsize=26)
        ax.set_ylabel("mape", fontsize=26)
        # ax.set_xscale('log')
        plt.savefig(
            f"./plots/results/{params['model']}_{params['f']}_{x_temp}_{conf['max_cov']}.pdf"
        )
        plt.tight_layout()
        # plt.show()
