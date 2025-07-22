import logging
from copy import copy
from functools import partial

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from main.plots import plot_extr_bw, plot_extr_hyp

from .estimations import corr, density_estimation, mape, rmse, rmspe
from .simulation import (DummyModel, random_gaussian_mixture_func,
                         random_GMM_samples, random_GP_func,
                         random_linear_func, random_matern_samples,
                         random_quadratic_func, random_thomas_samples,
                         random_uniform_samples, visualize_pattern)

log = logging.getLogger("__main__")


def run(
    conf,
    params,
    methods_list,
    hyp_params_dict,
):
    f_gens = {
        "linear": partial(random_linear_func, conf),
        "quadratic": partial(random_quadratic_func, conf),
        "GMM": partial(random_gaussian_mixture_func, conf),
        "GP": partial(random_GP_func, conf),
    }

    models = {"linear": DummyModel(), "boosting": GradientBoostingRegressor()}

    samples = {
        "GMM": random_GMM_samples,
        "Thomas": random_thomas_samples,
        "Matern": random_matern_samples,
    }

    params["f_gen"] = f_gens[params["f"]]
    params["model_gen"] = models[params["model"]]
    params["g_gen"] = partial(samples[params["samples"]], conf)

    conf_p = copy(conf)
    if params["p_style"] == "uniform":
        params["p_gen"] = partial(random_uniform_samples, conf, True)

    elif params["p_style"] == "GMM":

        conf_p["max_cov"] = 400
        params["p_gen"] = partial(samples["GMM"], conf_p)

    kf = (
        KFold(n_splits=params["n_splits"])
        if params["n_splits"] > 1
        else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    )

    for method in methods_list:
        for n_bw, _ in enumerate(method.bw_list):
            for n_hyp, _ in enumerate(method.hyperparams_list):
                method.test_errors_list[n_bw][n_hyp] = np.zeros(params["n_tests"])

    bw_estim_dict = {}
    for bw in hyp_params_dict["bw_list"]:
        bw_estim_dict[bw] = 0

    for n_test in trange(params["n_tests"]):
        test_gen_dict = {}
        test_gen_dict_p = {}
        g_estim_dict = {}
        p_estim_dict = {}

        test_gen_dict["f"] = params["f_gen"]()
        g_sample, g_estim_dict["g"] = params["g_gen"]()
        p_sample, test_gen_dict["p"] = params["p_gen"]()
        test_gen_dict["model"] = params["model_gen"]

        if n_test == 0:
            in_bounds = np.all((g_sample >= 0) & (g_sample <= conf["max_mu"]), axis=1)
            log.debug(f"Number of g_sample in bounds: {np.sum(in_bounds)}")
            visualize_pattern(g_sample, conf, params["samples"], alpha=0.7)
            visualize_pattern(p_sample, conf, "uniform", alpha=0.7)

        for n_fold, (train_idx, test_idx) in enumerate(kf.split(g_sample)):

            test_gen_dict["g_train"] = g_sample[train_idx]
            test_gen_dict["g_test"] = g_sample[test_idx]
            test_gen_dict["p_train"] = p_sample[train_idx]
            test_gen_dict["p_test"] = p_sample[test_idx]

            test_gen_dict_p["g_train"] = p_sample[train_idx]
            test_gen_dict_p["g_test"] = p_sample[test_idx]

            log.debug(f"test_gen_dict: {test_gen_dict}")

            params["model_gen"].fit(
                test_gen_dict["g_train"], test_gen_dict["f"](test_gen_dict["g_train"])
            )

            test_gen_dict["err"] = lambda X: np.log(
                np.abs(test_gen_dict["f"](X) - params["model_gen"].predict(X))
            )

            for bw in hyp_params_dict["bw_list"]:
                g_estim_dict[bw], bw_temp = density_estimation(
                    conf, hyp_params_dict, test_gen_dict, bw
                )
                if bw != "KL":
                    p_estim_dict[bw], bw_temp_p = density_estimation(
                        conf_p, hyp_params_dict, test_gen_dict_p, bw
                    )

                bw_estim_dict[bw] += bw_temp / (params["n_tests"] * params["n_splits"])

            log.debug(f"n_test: {n_test},\n test/train idx:{train_idx},\n {test_idx}\n")
            log.debug(f"g_estim_dict:\n {g_estim_dict}\n")

            for method in methods_list:
                log.debug(f"method.name = {method.name}")
                log.debug(f"method.bw_list = {method.bw_list}")
                for n_bw, bw in enumerate(method.bw_list):
                    test_gen_dict["g"] = g_estim_dict[bw]
                    if bw != "g":
                        test_gen_dict["p"] = p_estim_dict[bw]
                    for n_hyp, hyperparam in enumerate(method.hyperparams_list):
                        method.test_errors_list[n_bw][n_hyp][n_test] = (
                            method.test_errors_list[n_bw][n_hyp][n_test]
                            + method.single_test(
                                conf, test_gen_dict, hyperparam, hyp_params_dict
                            )
                            / params["n_splits"]
                        )
                log.debug(f"got test_errors_list: {method.test_errors_list}")

    proceed_metrics(conf, params, methods_list, hyp_params_dict, bw_estim_dict)

    return True


def proceed_metrics(conf, params, methods_list, hyp_params_dict, bw_estim_dict):
    y_method = [method for method in methods_list if method.name == "MCE_p"][0]
    x_methods_list = [method for method in methods_list if method.name != "MCE_p"]

    log.debug(
        f"\ny_method:{y_method.name}, new x_methods_list:{[method.name for method in x_methods_list]}\n"
    )

    y_err = np.exp(y_method.test_errors_list[0][0])
    log.debug(f"y_err {y_err}")
    for x_method in x_methods_list:
        best_indexes = []
        best_metrics_hyp = {
            "mape": [],
            "rmse": [],
            "rmspe": [],
            "corr": [],
            "mape_interval": [],
            "rmse_interval": [],
            "rmspe_interval": [],
            "corr_interval": [],
        }

        for n_bw, bw in enumerate(x_method.bw_list):
            for n_hyp, _ in enumerate(x_method.hyperparams_list):
                (
                    x_method.test_metrics_dict["mape"][n_bw][n_hyp],
                    x_method.test_metrics_dict["mape_interval"][n_bw][n_hyp],
                ) = mape(
                    np.exp(x_method.test_errors_list[n_bw][n_hyp]),
                    y_err,
                    hyp_params_dict["confidence"],
                )
                (
                    x_method.test_metrics_dict["rmse"][n_bw][n_hyp],
                    x_method.test_metrics_dict["rmse_interval"][n_bw][n_hyp],
                ) = rmse(
                    np.exp(x_method.test_errors_list[n_bw][n_hyp]),
                    y_err,
                    hyp_params_dict["confidence"],
                )
                (
                    x_method.test_metrics_dict["rmspe"][n_bw][n_hyp],
                    x_method.test_metrics_dict["rmspe_interval"][n_bw][n_hyp],
                ) = rmspe(np.exp(x_method.test_errors_list[n_bw][n_hyp]), y_err, 0.95)

                (
                    x_method.test_metrics_dict["corr"][n_bw][n_hyp],
                    x_method.test_metrics_dict["corr_interval"][n_bw][n_hyp],
                ) = corr(np.exp(x_method.test_errors_list[n_bw][n_hyp]), y_err, 0.95)
            log.debug(
                f"x_method:{x_method.name}, bw:{bw}\n test_metrics_dict[mape][bw]:\n{x_method.test_metrics_dict['mape'][n_bw]}\n"
            )

            best_index = np.argmin(x_method.test_metrics_dict["mape"][n_bw])
            log.debug(f"best_index: {best_index}")

            best_indexes += [best_index]
            best_metrics_hyp["mape"] += [
                x_method.test_metrics_dict["mape"][n_bw][best_index]
            ]
            best_metrics_hyp["mape_interval"] += [
                x_method.test_metrics_dict["mape_interval"][n_bw][best_index]
            ]
            best_metrics_hyp["rmse"] += [
                x_method.test_metrics_dict["rmse"][n_bw][best_index]
            ]

            best_metrics_hyp["rmse_interval"] += [
                x_method.test_metrics_dict["rmse_interval"][n_bw][best_index]
            ]
            best_metrics_hyp["rmspe"] += [
                x_method.test_metrics_dict["rmspe"][n_bw][best_index]
            ]

            best_metrics_hyp["rmspe_interval"] += [
                x_method.test_metrics_dict["rmspe_interval"][n_bw][best_index]
            ]
            best_metrics_hyp["corr"] += [
                x_method.test_metrics_dict["corr"][n_bw][best_index]
            ]

            best_metrics_hyp["corr_interval"] += [
                x_method.test_metrics_dict["corr_interval"][n_bw][best_index]
            ]
            log.debug(
                f"best_metrics_hyp[mape]:\n{best_metrics_hyp['mape']}\ninterval:\n{best_metrics_hyp['mape_interval']}\n"
            )

            plot_extr_hyp(conf, params, x_method, n_bw)

        log.debug(
            f"final best_metrics_hyp[mape]:\n{best_metrics_hyp['mape']},\n interval:\n {best_metrics_hyp['mape_interval']}\n"
        )

        best_index = np.argmin(best_metrics_hyp["mape"])
        log.debug(f"best index for bw_search: {best_index}")

        x_method.best_metrics_dict["mape"] += [best_metrics_hyp["mape"][best_index]]
        x_method.best_metrics_dict["mape_interval"] += [
            best_metrics_hyp["mape_interval"][best_index]
        ]
        x_method.best_metrics_dict["rmse"] += [best_metrics_hyp["rmse"][best_index]]
        x_method.best_metrics_dict["rmse_interval"] += [
            best_metrics_hyp["rmse_interval"][best_index]
        ]
        x_method.best_metrics_dict["rmspe"] += [best_metrics_hyp["rmspe"][best_index]]
        x_method.best_metrics_dict["rmspe_interval"] += [
            best_metrics_hyp["rmspe_interval"][best_index]
        ]
        x_method.best_metrics_dict["corr"] += [best_metrics_hyp["corr"][best_index]]
        x_method.best_metrics_dict["corr_interval"] += [
            best_metrics_hyp["corr_interval"][best_index]
        ]

        x_method.best_hyperparams_list += [
            x_method.hyperparams_list[best_indexes[best_index]]
        ]
        x_method.best_bw_list += [x_method.bw_list[best_index]]

        log.debug(
            f"adding best-best metrics and hyps for x_method: {x_method.name}:\nbest metrics_dict:\n{x_method.best_metrics_dict}\n"
        )
        log.debug(
            f"best_hyperparams_list:\n{x_method.best_hyperparams_list},\n best_bw_list:{x_method.best_bw_list}"
        )

        plot_extr_bw(conf, params, best_metrics_hyp["mape"], x_method)

    return True