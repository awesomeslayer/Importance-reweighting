import logging

import numpy as np
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
from tqdm import trange

from tests.metrics import mape, rmse
from tests.test import test

from .errors_init import errors_init
from .LSCV import KL_LSCV_find_bw


def errors_test(
    err_dict,
    err_hyp_dict,
    conf,
    params,
    hyperparams_dict,
    hyperparams_params,
    sizes,
    kf,
    i,
):
    log = logging.getLogger("__main__")

    gen_dict = {}
    gen_dict["f"] = params["f_gen"]()
    g_sample, gen_dict["g"] = params["g_gen"]()
    p_sample, gen_dict["p"] = params["p_gen"]()
    gen_dict["model"] = params["model_gen"]

    error, error_hyp = errors_init(params, sizes, 0)

    for train_idx, test_idx in kf.split(g_sample):
        log.debug(f"\ntrain_idx : {train_idx},\n test_tdx: {test_idx}\n")

        fill_gen_dict(
            conf,
            gen_dict,
            params,
            hyperparams_dict,
            hyperparams_params,
            g_sample,
            p_sample,
            train_idx,
            test_idx,
        )

        fill_errors(
            error,
            error_hyp,
            conf,
            hyperparams_dict,
            params,
            hyperparams_params,
            sizes,
            gen_dict,
        )

    log.debug(f"\nfinal error (summed for each fold) for one test num_{i}:\n {error}\n")
    log.debug(
        f"\nfinal hyperparams error (summed for each fold) for one test num_{i}:\n {error_hyp}\n"
    )

    fill_dicts(
        err_dict, err_hyp_dict, error, error_hyp, params, hyperparams_params, sizes
    )

    return True


def fill_gen_dict(
    conf,gen_dict, params, hyperparams_dict, hyperparams_params, g_sample, p_sample, train_idx, test_idx
):
    log = logging.getLogger("__main__")
    gen_dict["g_train"] = g_sample[train_idx]
    gen_dict["g_test"] = g_sample[test_idx]
    gen_dict["p_test"] = p_sample[test_idx]

    params["model_gen"].fit(gen_dict["g_train"], gen_dict["f"](gen_dict["g_train"]))

    gen_dict["err"] = lambda X: np.log(
        np.abs(gen_dict["f"](X) - params["model_gen"].predict(X))
    )

    if([s for s in params['x'] + params['x_hyp'] if "estim" in s or "reg" in s]):
        bw = hyperparams_dict["bandwidth"]
        if hyperparams_dict["bandwidth"] == 'KL_LSCV':
            bw = KL_LSCV_find_bw(conf, gen_dict["g_train"], gen_dict["p_test"], hyperparams_params["beta"])

        kde_sk = KernelDensity(kernel="gaussian", bandwidth=bw).fit(gen_dict["g_train"])

        if hyperparams_dict["bandwidth"] == "scott" or hyperparams_dict["bandwidth"] == "silverman":
            hyperparams_dict["bandwidth"] = kde_sk.bandwidth_
            log.debug(f"current for scott/silvername bandwidth is: {hyperparams_dict['bandwidth']}")
        
        gen_dict["g_estim"] = lambda X: kde_sk.score_samples(X)
        log.debug(f"\ng_estim : {gen_dict['g_estim']}\n")
    return True


def fill_errors(
    error,
    error_hyp,
    conf,
    hyperparams_dict,
    params,
    hyperparams_params,
    sizes,
    gen_dict,
):
    log = logging.getLogger("__main__")

    error_temp = test(
        conf,
        hyperparams_params,
        gen_dict,
        params["x"] + params["y"],
    )

    for x_temp in params["x"] + params["y"]:
        error[x_temp] = error[x_temp] + np.exp(error_temp[x_temp])
    
    log.debug(
        f"\ncurrent errors_dict for one test, summing for folds (no hyp methods):\n {error}\n"
    )

    for j in trange(sizes["max_size"]):
        hyperparams = {"kde_size": hyperparams_dict["bandwidth"]}
        x_estim_temp = []
        for x_temp in params["x_hyp"]:
            if j < sizes[x_temp]:
                hyperparams[x_temp] = hyperparams_dict[x_temp][j]
                x_estim_temp += [x_temp]

       
        error_hyp_temp = test(
            conf,
            hyperparams_params,
            gen_dict,
            x_estim_temp,
            hyperparams,
        )
        
        for x_temp in x_estim_temp:
            if x_temp in ['ISE_g_estim_clip, ISE_g_estim_clip_KL', 'ISE_g_clip']:
                if hyperparams_dict[x_temp][j] == "quantile":
                    hyperparams_dict[x_temp][j] == hyperparams[x_temp]
        

        for x_temp in x_estim_temp:
            error_hyp[x_temp][j] = error_hyp[x_temp][j] + np.exp(
                error_hyp_temp[x_temp]
            )

    log.debug(
        f"\ncurrent error for one test (sum for folds) with hyperparams:\n {error_hyp}\n"
    )

    return True


def fill_dicts(
    err_dict, err_hyp_dict, error, error_hyp, params, hyperparams_params, sizes
):
    log = logging.getLogger("__main__")

    for x_temp in params["x"] + params["y"]:
        err_dict[x_temp] += [error[x_temp] / params["n_splits"]]
    log.debug(
        f"\ntemp error_dict (average errors for folds) for one test:\n {err_dict}\n"
    )

    for x_temp in params["x_hyp"]:
        for j in range(sizes["max_size"]):
            if j < sizes[x_temp]:
                err_hyp_dict[x_temp][j] += [
                    error_hyp[x_temp][j] / params["n_splits"]
                ]
    log.debug(
        f"\nerror_dict_hyp (average errors for folds) for one test:\n {err_hyp_dict}\n"
    )
    
    return True


def count_metrics(params, hyperparams_dict, sizes, err_hyp_dict, err_dict):
    log = logging.getLogger("__main__")

    best_hyperparams = {}
    best_metrics_dict = {"mape": {}, "rmse": {}}

    y_err = np.array(err_dict[params["y"][0]])
    for x_temp in params["x"]:
        x_err = np.array(err_dict[x_temp])
        best_metrics_dict["mape"][x_temp] = mape(x_err, y_err)
        best_metrics_dict["rmse"][x_temp] = rmse(x_err, y_err)

    metrics_dict = {"mape": {}, "rmse": {}}
    for x_temp in params["x_hyp"]:
        metrics_dict["mape"][x_temp] = []
        metrics_dict["rmse"][x_temp] = []


    for x_temp in params["x_hyp"]:
        for i in range(sizes[x_temp]):
            x_err = np.array(err_hyp_dict[x_temp][i])
            metrics_dict["mape"][x_temp] += [mape(x_err, y_err)]
            metrics_dict["rmse"][x_temp] += [rmse(x_err, y_err)]

        best_hyperparams[x_temp] = hyperparams_dict[x_temp][
            metrics_dict["mape"][x_temp].index(min(metrics_dict["mape"][x_temp]))
        ]

        best_metrics_dict["mape"][x_temp] = metrics_dict["mape"][x_temp][
            metrics_dict["mape"][x_temp].index(min(metrics_dict["mape"][x_temp]))
        ]
        best_metrics_dict["rmse"][x_temp] = metrics_dict["rmse"][x_temp][
            metrics_dict["mape"][x_temp].index(min(metrics_dict["mape"][x_temp]))
        ]

    log.info(f"\nbest hyperparams:\n {best_hyperparams}\n")
    log.debug(f"\nmetrics_dict:\n {metrics_dict}\n")
    
    return best_metrics_dict, metrics_dict
