import logging
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange
from functools import partial

from .simulation import (
    DummyModel,
    random_gaussian_mixture_func,
    random_GMM_samples,
    random_GP_func,
    random_linear_func,
    random_uniform_samples,
)
from main.plots import plot_extr_bw, plot_extr_hyp, plot_KL_bw
from .estimations import density_estimation, mape, rmse

log = logging.getLogger("__main__")


def run(
    conf,
    params,
    methods_list,
    hyp_params_dict,
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

    kf = (
        KFold(n_splits=params["n_splits"])
        if params["n_splits"] > 1
        else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    )

    for method in methods_list:
        for n_bw, _ in enumerate(method.bw_list):
            for n_hyp, _ in enumerate(method.hyperparams_list):
                method.test_errors_list[n_bw][n_hyp] = np.zeros(params["n_tests"])

    for n_test in trange(params["n_tests"]):
        test_gen_dict = {}
        g_estim_dict = {}

        test_gen_dict["f"] = params["f_gen"]()
        g_sample, g_estim_dict["g"] = params["g_gen"]()
        p_sample, test_gen_dict["p"] = params["p_gen"]()
        test_gen_dict["model"] = params["model_gen"]

        for n_fold, (train_idx, test_idx) in enumerate(kf.split(g_sample)):

            test_gen_dict["g_train"] = g_sample[train_idx]
            test_gen_dict["g_test"] = g_sample[test_idx]
            test_gen_dict["p_train"] = p_sample[train_idx]
            test_gen_dict["p_test"] = p_sample[test_idx]

            if hyp_params_dict["KL_bw_plot"] == True and n_test == 0 and n_fold == 0:
                log.debug(f"for n_test: {n_test} and n_fold: {n_fold} getting plot:")
                plot_KL_bw(
                    conf,
                    test_gen_dict["g_train"],
                    test_gen_dict["p_train"],
                    hyp_params_dict["beta"],
                    hyp_params_dict["KL_enable"],
                    hyp_params_dict["estim_type"],
                    params,
                )

            log.debug(f"test_gen_dict: {test_gen_dict}")

            params["model_gen"].fit(
                test_gen_dict["g_train"], test_gen_dict["f"](test_gen_dict["g_train"])
            )

            test_gen_dict["err"] = lambda X: np.log(
                np.abs(test_gen_dict["f"](X) - params["model_gen"].predict(X))
            )

            for bw in hyp_params_dict["bw_list"]:
                g_estim_dict[bw] = density_estimation(
                    conf, hyp_params_dict, test_gen_dict, bw
                )

            log.debug(f"n_test: {n_test},\n test/train idx:{train_idx},\n {test_idx}\n")
            log.debug(f"g_estim_dict:\n {g_estim_dict}\n")

            for method in methods_list:
                log.debug(f"method.name = {method.name}")
                log.debug(f"method.bw_list = {method.bw_list}")
                for n_bw, bw in enumerate(method.bw_list):
                    test_gen_dict["g"] = g_estim_dict[bw]
                    for n_hyp, hyperparam in enumerate(method.hyperparams_list):
                        method.test_errors_list[n_bw][n_hyp][n_test] = (
                            method.test_errors_list[n_bw][n_hyp][n_test]
                            + method.single_test(
                                conf, test_gen_dict, hyperparam, hyp_params_dict
                            )
                            / params["n_splits"]
                        )
                log.debug(f"got test_errors_list: {method.test_errors_list}")
    proceed_metrics(conf, params, methods_list, hyp_params_dict["confidence"])

    return True


def proceed_metrics(conf, params, methods_list, confidence):

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
            "mape_interval": [],
            "rmse_interval": [],
        }

        for n_bw, bw in enumerate(x_method.bw_list):
            for n_hyp, _ in enumerate(x_method.hyperparams_list):
                (
                    x_method.test_metrics_dict["mape"][n_bw][n_hyp],
                    x_method.test_metrics_dict["mape_interval"][n_bw][n_hyp],
                ) = mape(
                    np.exp(x_method.test_errors_list[n_bw][n_hyp]), y_err, confidence
                )
                (
                    x_method.test_metrics_dict["rmse"][n_bw][n_hyp],
                    x_method.test_metrics_dict["rmse_interval"][n_bw][n_hyp],
                ) = rmse(
                    np.exp(x_method.test_errors_list[n_bw][n_hyp]), y_err, confidence
                )

            log.debug(
                f"x_method:{x_method.name}, bw:{bw}\n test_metrics_dict[mape][bw]:{x_method.test_metrics_dict['mape'][n_bw]}\n"
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

            log.debug(
                f"best_metrics_hyp[mape], interval: {best_metrics_hyp['mape']}, {best_metrics_hyp['mape_interval']}"
            )

            plot_extr_hyp(conf, params, x_method, n_bw)

        log.debug(
            f"final best_metrics_hyp[mape], interval:{best_metrics_hyp['mape']}, {best_metrics_hyp['mape_interval']}\n"
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

        x_method.best_hyperparams_list += [
            x_method.hyperparams_list[best_indexes[best_index]]
        ]
        x_method.best_bw_list += [x_method.bw_list[best_index]]

        log.debug(
            f"adding best-best metrics and hyps for x_method: {x_method.name}:best metrics_dict:\n{x_method.best_metrics_dict}\n"
        )
        log.debug(
            f"best_hyperparams_list:{x_method.best_hyperparams_list},\n best_bw_list:{x_method.best_bw_list}"
        )

        plot_extr_bw(conf, params, best_metrics_hyp["mape"], x_method)

    return True
