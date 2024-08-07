from copy import copy
from functools import partial
import logging
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange, tqdm

from plots.param_plots import bw_plot, extr_plots

from .errors import count_metrics, errors_test
from .errors_init import errors_init, find_sizes
from .simulation import (
    DummyModel,
    random_gaussian_mixture_func,
    random_GMM_samples,
    random_GP_func,
    random_linear_func,
    random_uniform_samples,
)


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

    best_metrics_dict = {"mape": {}, "rmse": {}}
    for x_temp in params["x"] + params["x_hyp"]:
        best_metrics_dict["mape"][x_temp] = []
        best_metrics_dict["rmse"][x_temp] = []

    log.debug(f"\nbest_metrics_dict for max_cov = {conf['max_cov']}init: {best_metrics_dict}\n")

    bw_list = []
    for bw in tqdm(hyperparams_params["bw_list"]):
        log.debug(f"\ncurrent bandwidth = {bw}\n")
        bw_extr_metrics_dict = {}

        params_temp = copy(params)

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
            params_temp["x"] = []
            params_temp["x_hyp"] = []

            step = 0
            for x_temp in list(set(params["x"]) & set(["ISE_g_estim"])):    
                params_temp["x"] += [x_temp + '_KL']
            
            for x_temp in list(set(params["x_hyp"]) & set(["ISE_g_estim_clip", "ISE_g_reg_uniform", "ISE_g_reg_degree"])):
                
                hyperparams_dict[x_temp + '_KL'] = hyperparams_dict[x_temp]
                params_temp["x_hyp"] += [x_temp + '_KL']
               
            for x_temp in params_temp["x"] + params_temp["x_hyp"]:
                best_metrics_dict["mape"][x_temp] = []
                best_metrics_dict["rmse"][x_temp] = []
            log.debug(f"\nbest_metrics_dict with KL init: {best_metrics_dict}\n")

        hyperparams_dict["bandwidth"] = bw
        log.debug(f"\nparams_temp = {params_temp}\n")
        log.debug(f"\nhyperparams_dict_temp = {hyperparams_dict}\n")

        sizes = find_sizes(params_temp, hyperparams_dict)
        log.debug(f"\n temp_sizes = {sizes}\n")

        err_test_dict, err_test_hyp_dict = errors_init(
            params_temp, sizes, list()
        )

        log.debug(f"\nerr_test_dict init fow bw={bw}  : {err_test_dict}\n")
        log.debug(f"\nerr_test_hyp_dict init for bw={bw} : {err_test_hyp_dict}\n")

        for i in trange(params_temp["n_tests"]):
            errors_test(
                err_test_dict,
                err_test_hyp_dict,
                conf,
                params_temp,
                hyperparams_dict,
                hyperparams_params,
                sizes,
                kf,
                i,
            )

        bw_extr_metrics_dict, bw_metrics_dict = count_metrics(
            params_temp,
            hyperparams_dict,
            sizes,
            err_test_hyp_dict,
            err_test_dict,
        )

        if bw != "KL_LSCV":
            bw_list += [hyperparams_dict["bandwidth"]]

        for x_temp in params_temp["x"] + params_temp["x_hyp"]:
            best_metrics_dict["mape"][x_temp] += [bw_extr_metrics_dict["mape"][x_temp]]
            best_metrics_dict["rmse"][x_temp] += [bw_extr_metrics_dict["rmse"][x_temp]]

        
        extr_plots(conf, params_temp, bw_metrics_dict, hyperparams_dict, bw)
        
        log.info(
            f"\n for bw = {bw} extr_metrics_dict for max_cov {conf['max_cov']}= \n{bw_extr_metrics_dict}\n"
        )

    log.debug(f"\nbest_metrics_dict lists for bw: {best_metrics_dict}\n")

    best_metrics_dict = bw_plot(conf, params, bw_list, best_metrics_dict)
    log.debug(f"\nbest_metrics_dict minimums for bw : {best_metrics_dict}\n")

    return best_metrics_dict
