from functools import partial
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, ShuffleSplit
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from sklearn.neighbors import KernelDensity
import logging
from copy import copy

from tests.metrics import mape, rmse
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

    kf = (
        KFold(n_splits=params["n_splits"])
        if params["n_splits"] > 1
        else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    )

    #get sizes for hyperparams_search
    sizes = find_sizes(params, hyperparams_dict)    
    logging.info(f"\n sizes = {sizes}\n")
    #init errors dictionaries
    err_dict, err_hyp_dict = errors_init(params, sizes, list())

    logging.info(f"\nerr_dict init = {err_dict}\n")
    logging.info(f"\nerr_hyp_dict init = {err_hyp_dict}\n")

    #test all methods, get errors dictionaries (for each method and each param on same generations (same folds) for each tests)
    for i in trange(params["n_tests"]):
        errors_test(err_dict, err_hyp_dict, conf, params, hyperparams_dict, hyperparams_params, sizes, kf, i)
    
    #if needed find best hyperparams and get best_mape/rmse dictionaries with them
    if hyperparams_params["grid_flag"]:
        best_metrics_dict, metrics_dict = find_best_params(
            params, hyperparams_dict, sizes, err_hyp_dict, err_dict
        )
        logging.info(f"\nmetrics_dict:\n {metrics_dict}\n")
        logging.info(f"\ncheck limits for regularisation:\ndelta_mape 0 ?= {metrics_dict['mape']['ISE_g_regular'][0]} - {best_metrics_dict['mape']['ISE_g_estim'][0]}\n")
    
        extr_plots(conf, params, metrics_dict, hyperparams_dict)
    #else use only first hyperparam from each method config and count errors with them
    else:
        y_err = np.array(err_dict[params["y"][0]])
        for x_temp in params["x"] + params["x_hyp"] + params["y"]:
            x_err = np.array(err_dict[x_temp])
            best_metrics_dict["mape"][x_temp] += [mape(x_err, y_err)]
            best_metrics_dict["rmse"][x_temp] += [rmse(x_err, y_err)]

    logging.info(f"\nbest_metrics_dict = \n{best_metrics_dict}\n")
    return best_metrics_dict

def find_sizes(params, hyperparams_dict):
    sizes = dict()
    max_size = 0
    for x_temp in params["x_hyp"]:
        sizes[x_temp] = len(hyperparams_dict[x_temp])
        if len(hyperparams_dict[x_temp]) > max_size:
            max_size = len(hyperparams_dict[x_temp])
    
    sizes["max_size"] = max_size
    return sizes

def errors_init(params, sizes, type):
    err_dict = {}
    for x_temp in params["y"] + params["x"]:
        err_dict[x_temp] = copy(type)

    err_hyp_dict = {}
    for x_temp in params["x_hyp"]:
        err_hyp_dict[x_temp] = []
        for i in range(sizes[x_temp]):
            err_hyp_dict[x_temp].append(copy(type))

    return err_dict, err_hyp_dict


def errors_test(err_dict, err_hyp_dict, conf, params, hyperparams_dict, hyperparams_params, sizes, kf, i):
    
    gen_dict = {}
    gen_dict["f"] = params["f_gen"]()
    g_sample, gen_dict["g"] = params["g_gen"]()
    p_sample, gen_dict["p"] = params["p_gen"]()
    
    error, error_hyp = errors_init(params, sizes, 0)

    for train_idx, test_idx in kf.split(g_sample):

        logging.info(f"\ntrain_idx : {train_idx},\n test_tdx: {test_idx}\n")
        gen_dict["g_train"] = g_sample[train_idx]
        gen_dict["g_test"] = g_sample[test_idx]
        gen_dict["p_test"] = p_sample[test_idx]
        
        params["model_gen"].fit(gen_dict["g_train"], gen_dict["f"](gen_dict["g_train"]))

        gen_dict["err"] = lambda X: np.abs(gen_dict["f"](X) - params["model_gen"].predict(X))

        kde_list = [
            "ISE_g_estim",
            "ISE_g_regular",
            "ISE_g_estim_clip",
        ]
        if [i for i in params["x"] + params["x_hyp"] + params["y"] if i in kde_list]:
            kde_sk = KernelDensity(
                kernel="gaussian", bandwidth=hyperparams_dict["kde_size"][0]
            ).fit(gen_dict["g_train"])
            gen_dict["g_estim"] = lambda X: np.exp(kde_sk.score_samples(X))

        logging.info(f"\ng_estim : {gen_dict['g_estim']}\n")    
        # work without hyperparams:
        error_temp = test(
                    conf,
                    hyperparams_params,
                    gen_dict,
                    params["x"] + params["y"],
                )
         
        for x_temp in params["x"] + params["y"]:
            error[x_temp] = error[x_temp] + error_temp[x_temp]
        logging.info(f"\ncurrent error without hyperparams:\n {error}\n")
        
        if hyperparams_params["grid_flag"]:
            for j in trange(sizes['max_size']):
                hyperparams = {"kde_size": hyperparams_dict["kde_size"][0]}
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
                    error_hyp[x_temp][j] = error_hyp[x_temp][j] + error_hyp_temp[x_temp]
                logging.info(f"\ncurrent error with hyperparams:\n {error_hyp}\n")    
        else:
            logging.info("\ngrid flag is off, no hyp-search:\n")
            error_hyp_temp = test(
                            conf,
                            hyperparams_params,
                            gen_dict,
                            params["x_hyp"],
                            best_hyperparams = {
                                "kde_size": hyperparams_dict["kde_size"][0],
                                "n_slices": hyperparams_dict["Mandoline"][0],
                                "ISE_g_regular": hyperparams_dict["ISE_g_regular"][0],
                                "ISE_g_clip": hyperparams_dict["ISE_g_clip"][0],
                                "ISE_g_estim_clip": hyperparams_dict["ISE_g_clip"][0],})
            
            for x_temp in params["x_hyp"]:
                error[x_temp] = error[x_temp] + error_hyp_temp[x_temp]
            
            logging.info(f"\ncurrent error for fixed hyperparams:\n {error}\n")

    logging.info(f"\nfinal error for one test num_{i}:\n {error}\n")
    logging.info(f"\nfinal hyperparams error for one test num_{i}:\n {error_hyp}\n")

    for x_temp in params["x"] + params["y"]:
        error_temp = error[x_temp]/params["n_splits"]
        logging.info(f"\nerror_temp in default for {x_temp} = {error_temp}\n")
        err_dict[x_temp] += [error_temp]
        logging.info(f"\ntemp error_dict:\n {err_dict}\n")
        
    if hyperparams_params["grid_flag"]:
        for x_temp in params["x_hyp"]:
            for j in range(sizes['max_size']):
                if j < sizes[x_temp]:
                    error_temp = error_hyp[x_temp][j]/params["n_splits"]
                    logging.info(f"\nerror_temp hyperparams for {x_temp} = {error_temp}\n")
                    err_hyp_dict[x_temp][j] += [error_temp]  
            logging.info(f"\nerror_dict_hyp :\n {err_hyp_dict}\n")
                    
    else:
        logging.info("\n for grid_flag = False:\n")
        for x_temp in params["x_hyp"]:
            error_temp = error[x_temp]/params["n_splits"]
            logging.info(f"\nerror_temp hyp:\n {error_temp}\n")
            err_dict[x_temp] += [error[x_temp]/params["n_splits"]]  

    logging.info(f"\ntest_{i}, current err_dict:\n{err_dict}\n")
    logging.info(f"\ntest_{i}, current err_hyp_dict:\n{err_hyp_dict}\n")
    return True

def find_best_params(params, hyperparams_dict, sizes, err_hyp_dict, err_dict):
    best_hyperparams = {}
    best_metrics_dict = {"mape": {}, "rmse": {}}
    metrics_dict = {"mape": {}, "rmse": {}}

    for x_temp in params["x_hyp"]:
        metrics_dict["mape"][x_temp] = []
        metrics_dict["rmse"][x_temp] = []
    
    for x_temp in params["x"]:
        best_metrics_dict["mape"][x_temp] = []
        best_metrics_dict["rmse"][x_temp] = []

    y_err = np.array(err_dict[params["y"][0]])
    for x_temp in params["x"]:
        x_err = np.array(err_dict[x_temp])
        best_metrics_dict["mape"][x_temp] += [mape(x_err, y_err)]
        best_metrics_dict["rmse"][x_temp] += [rmse(x_err, y_err)]
    
    for x_temp in params["x_hyp"]:
        for i in range(sizes[x_temp]):
            x_err = np.array(err_hyp_dict[x_temp][i])
            metrics_dict["mape"][x_temp] += [mape(x_err, y_err)]
            metrics_dict["rmse"][x_temp] += [rmse(x_err, y_err)]

        best_hyperparams[x_temp] = hyperparams_dict[x_temp][
            metrics_dict["mape"][x_temp].index(
                min(metrics_dict["mape"][x_temp])
            )
        ]

        best_metrics_dict["mape"][x_temp] = [metrics_dict["mape"][x_temp][
            metrics_dict["mape"][x_temp].index(
                min(metrics_dict["mape"][x_temp])
            )
        ]]
        best_metrics_dict["rmse"][x_temp] = [metrics_dict["rmse"][x_temp][
            metrics_dict["mape"][x_temp].index(
                min(metrics_dict["mape"][x_temp])
            )
        ]]

    logging.info(f"\nbest hyperparams:\n {best_hyperparams}\n")
    return best_metrics_dict, metrics_dict


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
        #ax.set_xscale('log')
        plt.savefig(
            f"./plots/results/{params['model']}_{params['f']}_{x_temp}_{conf['max_cov']}.pdf"
        )
        plt.tight_layout()
        # plt.show()
