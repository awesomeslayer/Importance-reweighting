import logging
from copy import copy
from functools import partial

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from plots.param_plots import bw_plot, extr_plots

from .errors import count_metrics, errors_test
from .errors_init import errors_init, find_sizes
from .simulation import (DummyModel, random_gaussian_mixture_func,
                         random_GMM_samples, random_GP_func,
                         random_linear_func, random_uniform_samples)


def run(
    conf,
    params,
    hyperparams_params,
    hyperparams_dict,
):
    log = logging.getLogger("__main__")

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

    kf = (
        KFold(n_splits=params["n_splits"])
        if params["n_splits"] > 1
        else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    )

    # get sizes for hyperparams_search
    sizes = find_sizes(params, hyperparams_dict)
    log.debug(f"\n sizes = {sizes}\n")
    # init errors dictionaries

    best_metrics_dict = {"mape": {}, "rmse": {}}
    for x_temp in params["x"] + params["x_hyp"]:
        best_metrics_dict["mape"][x_temp] = []
        best_metrics_dict["rmse"][x_temp] = []

    log.debug(f"best_metrics_dict with init: {best_metrics_dict}")

    bw_list = []
    for bw in hyperparams_params["bw_list"]:
        params_temp = copy(params)
        extr_metrics_dict = {}

        step = 0
        if bw != "KL_LSCV":
            if step != 0:
                params_temp["x"] = list(set(params["x"]) & set(["ISE_g_estim"]))
                params_temp["x_hyp"] = list(
                    set(params["x_hyp"])
                    & set(["ISE_g_estim_clip", "ISE_g_reg_uniform", "ISE_g_reg_degree"])
                )
            step = step + 1
        else:
            step = 0
            params_temp["x"] = []
            params_temp["x_hyp"] = [
                "ISE_g_estim_clip_KL",
                "ISE_g_reg_uniform_KL",
                "ISE_g_estim_KL",
                "ISE_g_reg_degree_KL",
            ]
            for x_temp in params_temp["x_hyp"]:
                best_metrics_dict["mape"][x_temp] = []
                best_metrics_dict["rmse"][x_temp] = []
            log.debug(f"best_metrics_dict with KL init: {best_metrics_dict}")

        err_dict, err_hyp_dict = errors_init(
            params_temp, sizes, list(), hyperparams_params["grid_flag"]
        )

        log.debug(f"\nerr_dict init fow bw={bw}  : {err_dict}\n")
        log.debug(f"\nerr_hyp_dict init for bw={bw} : {err_hyp_dict}\n")

        # test all methods, get errors dictionaries (for each method and each param on same generations (same folds) for each tests)
        hyperparams_dict["bandwidth"] = bw
        hyperparams_dict["beta"] = hyperparams_params[
            "beta"
        ]  # in future may be plots for betas, and edit for list...
        for i in trange(params_temp["n_tests"]):
            errors_test(
                err_dict,
                err_hyp_dict,
                conf,
                params_temp,
                hyperparams_dict,
                hyperparams_params,
                sizes,
                kf,
                i,
            )

        bw_list += [hyperparams_dict["bandwidth"]]
        # if needed find best hyperparams and get best_mape/rmse dictionaries with them
        extr_metrics_dict, metrics_dict = count_metrics(
            params_temp,
            hyperparams_dict,
            sizes,
            err_hyp_dict,
            err_dict,
            hyperparams_params["grid_flag"],
        )
        for x_temp in params_temp["x"] + params_temp["x_hyp"]:
            best_metrics_dict["mape"][x_temp] += [extr_metrics_dict["mape"][x_temp]]
            best_metrics_dict["rmse"][x_temp] += [extr_metrics_dict["rmse"][x_temp]]

        if hyperparams_params["grid_flag"]:
            extr_plots(conf, params_temp, metrics_dict, hyperparams_dict, bw)
        log.info(
            f"\n for bw = {bw} extr_metrics_dict for max_cov {conf['max_cov']}= \n{extr_metrics_dict}\n"
        )

    log.debug(f"best_metrics_dict lists for bw: {best_metrics_dict}")

    best_metrics_dict = bw_plot(conf, params, bw_list, best_metrics_dict)
    log.debug(f"best_metrics_dict minimums for bw : {best_metrics_dict}")

    return best_metrics_dict
