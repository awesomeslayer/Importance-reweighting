from functools import partial

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from tqdm import trange

from plots.extr_plots import extr_plots

from .errors import count_metrics, errors_test
from .errors_init import errors_init, find_sizes
from .simulation import (DummyModel, random_gaussian_mixture_func,
                         random_GMM_samples, random_GP_func,
                         random_linear_func, random_uniform_samples)


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
    log=None,
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

    # get sizes for hyperparams_search
    sizes = find_sizes(params, hyperparams_dict)
    log.debug(f"\n sizes = {sizes}\n")
    # init errors dictionaries
    err_dict, err_hyp_dict = errors_init(
        params, sizes, list(), hyperparams_params["grid_flag"]
    )

    log.debug(f"\nerr_dict init = {err_dict}\n")
    log.debug(f"\nerr_hyp_dict init = {err_hyp_dict}\n")

    # test all methods, get errors dictionaries (for each method and each param on same generations (same folds) for each tests)
    for i in trange(params["n_tests"]):
        errors_test(
            err_dict,
            err_hyp_dict,
            conf,
            params,
            hyperparams_dict,
            hyperparams_params,
            sizes,
            kf,
            i,
            log,
        )

    # if needed find best hyperparams and get best_mape/rmse dictionaries with them
    best_metrics_dict, metrics_dict = count_metrics(
        params,
        hyperparams_dict,
        sizes,
        err_hyp_dict,
        err_dict,
        hyperparams_params["grid_flag"],
        log,
    )
    if hyperparams_params["grid_flag"]:
        extr_plots(conf, params, metrics_dict, hyperparams_dict, params["log_flag"])
    log.info(
        f"\nbest_metrics_dict for max_cov {conf['max_cov']}= \n{best_metrics_dict}\n"
    )

    return best_metrics_dict
