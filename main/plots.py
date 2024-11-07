import matplotlib.pyplot as plt
import logging
import numpy as np
from functools import partial
from tqdm import trange, tqdm
from scipy.stats import kstest
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import gaussian_kde

from sklearn.neighbors import KernelDensity

from KL_divergence_estimators.knn_divergence import (
    naive_estimator,
    scipy_estimator,
    skl_efficient,
    skl_estimator,
)
from source.KL_LSCV import squared_error
from source.LCF import (
    test_LCF,
    estimate_lcf,
    estimate_point_counts,
    compute_normalized_auc,
    plot_lcf,
)

from source.simulation import (
    DummyModel,
    random_gaussian_mixture_func,
    random_GMM_samples,
    random_GP_func,
    random_linear_func,
    random_uniform_samples,
    random_thomas_samples,
    random_matern_samples
)

log = logging.getLogger("__main__")
log.setLevel(logging.DEBUG)

def plot_cov_KL_estim(conf, params, KL_estim_list=["naive", "scipy", "skl", "skl_ef"]):
    log.debug(f"KL_estim_list = {KL_estim_list}")
    samples = {'GMM' : random_GMM_samples, 'Thomas' : random_thomas_samples}

    KL_estim_func_dict = {
        "naive": lambda g, p: naive_estimator(g, p, k=5),
        "scipy": lambda g, p: scipy_estimator(g, p, k=5),
        "skl": lambda g, p: skl_estimator(g, p, k=5),
        "skl_ef": lambda g, p: skl_efficient(g, p, k=5),
    }

    KL_estim_dict = {}
    for key in KL_estim_list:
        KL_estim_dict[key] = np.zeros(len(params["max_cov_list"]))

    log.debug(f"KL_estim(max_cov) plot:")
    for j, cov in tqdm(enumerate(params["max_cov_list"])):
        conf["max_cov"] = cov

        
        g_gen = partial(samples[params['samples']], conf)
        p_gen = partial(random_uniform_samples, conf, True)

        log.debug(f"cov = {cov}:")
        for _ in trange(params["n_tests"]):
            g_sample, _ = g_gen()
            p_sample, _ = p_gen()

            for key in KL_estim_list:
                KL_estim_dict[key][j] = (
                    KL_estim_dict[key][j]
                    + KL_estim_func_dict[key](g_sample, p_sample) / params["n_tests"]
                )

    fig, ax = plt.subplots(figsize=(12, 12))
    for key in KL_estim_dict:
        ax.plot(
            params["max_cov_list"],
            KL_estim_dict[key],
            label=f"{key}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("KL_estimation between u and g", fontsize=26)
    plt.tight_layout()
    plt.savefig(
        f"./main/results/KL_plots/{params['model']}_{params['f']}/KL_estim(max_cov).pdf"
    )
    return True

def energy_distance_2d(sample1, sample2):
    dist_within_sample1 = cdist(sample1, sample1, 'euclidean')
    dist_within_sample2 = cdist(sample2, sample2, 'euclidean')
    dist_between_samples = cdist(sample1, sample2, 'euclidean')
    
    n = len(sample1)
    m = len(sample2)
    
    term1 = np.sum(dist_within_sample1) / (n * (n - 1))
    term2 = np.sum(dist_within_sample2) / (m * (m - 1))
    term3 = np.sum(dist_between_samples) / (n * m)
    
    return 2 * term3 - term1 - term2

def plot_cov_KS(conf, params, n_tests=5):
    KS_dict = {"p": [], "g": [], "2d": {"p": [], "g": []}}
    samples = {'GMM' : random_GMM_samples, 'Thomas' : random_thomas_samples, 'Matern' : random_matern_samples}

    for key in KS_dict:
        if key != "2d":
            KS_dict[key] = np.zeros((2, len(params["max_cov_list"]))) 
        else:
            KS_dict[key]["p"] = np.zeros(len(params["max_cov_list"])) 
            KS_dict[key]["g"] = np.zeros(len(params["max_cov_list"]))  
    
    log.debug(f"KS(max_cov) plot for single ax:")
    
    for j, cov in tqdm(enumerate(params["max_cov_list"]), total=len(params["max_cov_list"])):
        conf["max_cov"] = cov
        g_gen = partial(samples[params['samples']], conf)
        p_gen = partial(random_uniform_samples, conf, True)  
        
        log.debug(f"cov = {cov}:")
        
        cumulative_ks_p_lat = 0
        cumulative_ks_p_long = 0
        cumulative_ks_g_lat = 0
        cumulative_ks_g_long = 0
        cumulative_ks_2d_p = 0
        cumulative_ks_2d_g = 0
        
        for i in trange(n_tests, desc=f"Tests for cov={cov}"):
            g_sample, _ = g_gen()
            p_sample, _ = p_gen()

            lat_g = np.sort(g_sample[:, 0])
            long_g = np.sort(g_sample[:, 1])
            lat_p = np.sort(p_sample[:, 0])
            long_p = np.sort(p_sample[:, 1])
            
            ks_stat_g_lat, _ = kstest(lat_g, 'uniform', args=(np.min(lat_g), np.ptp(lat_g)))
            ks_stat_g_long, _ = kstest(long_g, 'uniform', args=(np.min(long_g), np.ptp(long_g)))
            ks_stat_p_lat, _ = kstest(lat_p, 'uniform', args=(np.min(lat_p), np.ptp(lat_p)))
            ks_stat_p_long, _ = kstest(long_p, 'uniform', args=(np.min(long_p), np.ptp(long_p)))

            cumulative_ks_g_lat += ks_stat_g_lat
            cumulative_ks_g_long += ks_stat_g_long
            cumulative_ks_p_lat += ks_stat_p_lat
            cumulative_ks_p_long += ks_stat_p_long

            cumulative_ks_2d_p += energy_distance_2d(p_sample, np.random.uniform(size=(len(p_sample), 2)) * 100)
            cumulative_ks_2d_g += energy_distance_2d(g_sample, np.random.uniform(size=(len(g_sample), 2)) * 100)

        KS_dict["g"][0, j] = cumulative_ks_g_lat / n_tests  
        KS_dict["g"][1, j] = cumulative_ks_g_long / n_tests  
        KS_dict["p"][0, j] = cumulative_ks_p_lat / n_tests 
        KS_dict["p"][1, j] = cumulative_ks_p_long / n_tests  

        KS_dict["2d"]["p"][j] = cumulative_ks_2d_p / n_tests
        KS_dict["2d"]["g"][j] = cumulative_ks_2d_g / n_tests
    
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(params["max_cov_list"], KS_dict["p"][0], label="p_sample Latitude", marker='o', color='blue')
    plt.plot(params["max_cov_list"], KS_dict["g"][0], label="g_sample Latitude", marker='o', color='orange')
    plt.xlabel("max_cov")
    plt.ylabel("Average KS Statistic (Latitude)")
    plt.title("Average KS Test for Latitude")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(params["max_cov_list"], KS_dict["p"][1], label="p_sample Longitude", marker='o', color='green')
    plt.plot(params["max_cov_list"], KS_dict["g"][1], label="g_sample Longitude", marker='o', color='red')
    plt.xlabel("max_cov")
    plt.ylabel("Average KS Statistic (Longitude)")
    plt.title("Average KS Test for Longitude")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(params["max_cov_list"], KS_dict["2d"]["p"], label="p_sample 2D KS", marker='o', color='purple')
    plt.plot(params["max_cov_list"], KS_dict["2d"]["g"], label="g_sample 2D KS", marker='o', color='brown')
    plt.xlabel("max_cov")
    plt.ylabel("Average Energy Distance (2D)")
    plt.title("Average 2D KS Test")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./main/results/KS_plots/{params['samples']}/KS(max_cov)_2D.pdf")
    plt.show()
    
    return True

def plot_cov_LCF(conf, params, n_tests=5):
    rmax = 0.188*conf['max_mu']
    r_values = np.arange(0, rmax, 0.1)
    test_LCF()
    samples = {'GMM' : random_GMM_samples, 'Thomas' : random_thomas_samples, 'Matern' : random_matern_samples}

    LCF_dict = {"p": [], "g": []}
    for key in LCF_dict:
        LCF_dict[key] = np.zeros(len(params["max_cov_list"]))

    log.debug(f"LCF(max_cov) plot:")
    for j, cov in tqdm(enumerate(params["max_cov_list"])):
        conf["max_cov"] = cov
        g_gen = partial(samples[params['samples']], conf)
        p_gen = partial(random_uniform_samples, conf, True)
        log.debug(f"cov = {cov}:")
        for i in trange(n_tests):
            g_sample, _ = g_gen()
            p_sample, _ = p_gen()

            p_est_df = estimate_point_counts(p_sample, r_values)
            g_est_df = estimate_point_counts(g_sample, r_values)

            p_lcf_df = estimate_lcf(p_est_df, r=r_values)
            g_lcf_df = estimate_lcf(g_est_df, r=r_values)

            AUC_p = compute_normalized_auc(p_lcf_df, r_values, rmin = rmax/10, rmax = rmax)
            AUC_g = compute_normalized_auc(g_lcf_df, r_values, rmin = rmax/10, rmax = rmax)

            plot_lcf(
                g_lcf_df, title=f"LCF_g_sample_cov({cov})_{i}", AUC=AUC_g, dir="/all"
            )

            plot_lcf(
                p_lcf_df, title=f"LCF_p_sample_cov({cov})_{i}", AUC=AUC_p, dir="/all"
            )

            for key, LCF_AUC in zip(LCF_dict, (AUC_p, AUC_g)):
                LCF_dict[key][j] = LCF_dict[key][j] + LCF_AUC / n_tests

    fig, ax = plt.subplots(figsize=(12, 12))
    for key in LCF_dict:
        ax.plot(
            params["max_cov_list"],
            LCF_dict[key],
            label=f"{key}",
        )
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("LCF_AUC_average", fontsize=26)
    plt.title(f"n_test = {n_tests}")
    plt.tight_layout()
    plt.savefig(f"./main/results/LCF_plots/{params['samples']}_LCF(max_cov).pdf")
    return True

# Modifying the function to add h_kl_avgs that tracks the h value corresponding to the minimum KL for each cov

def plot_cov_bw(
    conf,
    params,
    hyp_params_dict,
    h_list=np.linspace(0.01, 20, 300),
):
    log.info(f"Start cov bw plot")
    f_gens = {
        "linear": partial(random_linear_func, conf),
        "GMM": partial(random_gaussian_mixture_func, conf),
        "GP": partial(random_GP_func, conf),
    }

    models = {"linear": "DummyModel", "boosting": GradientBoostingRegressor()}
    samples = {'GMM' : random_GMM_samples, 'Thomas' : random_thomas_samples, 'Matern' : random_matern_samples}

    h_scott_avgs = []
    h_kl_avgs = []
    for cov in params['max_cov_list']:
        conf["max_cov"] = cov
        log.info(f"counting for cov = {cov}")
        
        test_gen_dict = {}
        g_estim_dict = {}

        params["f_gen"] = f_gens[params["f"]]
        params["model_gen"] = models[params["model"]]
        params["g_gen"] = partial(samples[params['samples']], conf)
        params["p_gen"] = partial(random_uniform_samples, conf, True)

        kl_sums = np.zeros(len(h_list))
        h_temps = []
        h_kl_min_per_test = []

        for i in range(params['n_tests']):
            test_gen_dict["f"] = params["f_gen"]()
            g_sample, g_estim_dict["g"] = params["g_gen"]()
            p_sample, test_gen_dict["p"] = params["p_gen"]()
            test_gen_dict["model"] = params["model_gen"]

            kf = (
                KFold(n_splits=params["n_splits"])
                if params["n_splits"] > 1
                else ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
            )

            for _, (train_idx, test_idx) in enumerate(kf.split(g_sample)):
                test_gen_dict["g_train"] = g_sample[train_idx]
                test_gen_dict["g_test"] = g_sample[test_idx]
                test_gen_dict["p_train"] = p_sample[train_idx]
                test_gen_dict["p_test"] = p_sample[test_idx]
                
                # Find average h referring to Scott bandwidth:
                if hyp_params_dict["estim_type"] == "sklearn":
                    kde = KernelDensity(kernel="gaussian", bandwidth='scott').fit(
                        test_gen_dict["g_train"]
                    )
                    h_temp = kde.bandwidth_
                elif hyp_params_dict["estim_type"] == "scipy":
                    kde = gaussian_kde(test_gen_dict["g_train"].T, bw_method='scott')
                    h_temp = kde.covariance_factor()
                
                h_temps.append(h_temp)
                
                kl_list = []
                log.info(f"for test = {i}")
                for h in h_list:
                    kl = squared_error(h, conf, g_sample, p_sample, hyp_params_dict['beta'], hyp_params_dict['KL_enable'], hyp_params_dict['estim_type'])
                    kl_list.append(kl)

                kl_sums += np.array(kl_list)

                # Get the h value that minimizes the KL for this test
                h_kl_min_per_test.append(h_list[np.argmin(kl_list)])

        # Find average h that minimizes KL
        kl_avg = kl_sums / params['n_tests']
        h_scott_avg = np.mean(h_temps)
        h_kl_avg = np.mean(h_kl_min_per_test)

        h_scott_avgs.append(h_scott_avg)
        h_kl_avgs.append(h_kl_avg)
        
        log.info(f"h_kl_avg = {h_kl_avg}, h_scott_avg = {h_scott_avg}")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(h_list, kl_avg, label="Average KL")
        plt.legend(fontsize=26)
        plt.title(f"KL(h) for {conf['max_cov']}", fontsize=20)
        ax.set_xlabel("h", fontsize=26)
        ax.set_ylabel("Average KL", fontsize=26)
        plt.tight_layout()
        plt.savefig(
            f"./main/results/KL_plots/{params['model']}_{params['f']}/cov_{conf['max_cov']}.pdf"
        )
    
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.plot(params['max_cov_list'], h_scott_avgs, label="h_scott_average", marker='o')
    ax.plot(params['max_cov_list'], h_kl_avgs, label="h_kl_average", marker='x')
    plt.legend(fontsize=26)
    plt.title(f"h_scott_average and h_kl_average vs max_cov_list", fontsize=20)
    ax.set_xlabel("max_cov_list", fontsize=26)
    ax.set_ylabel("Bandwidth", fontsize=26)
    plt.tight_layout()
    plt.savefig(f"./main/results/KL_plots/h_scott_and_h_kl_vs_cov.pdf")
    
    log.info(f"End plotting cov_bw")
    return True


def plot_extr_hyp(conf, params, x_method, n_bw):

    if len(x_method.hyperparams_list) > 1:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(
            x_method.hyperparams_list,
            x_method.test_metrics_dict["mape"][n_bw],
            label=f"{x_method.name}",
        )
        plt.legend(fontsize=26)
        plt.title(f"max_cov{conf['max_cov']}_bw{x_method.bw_list[n_bw]}")
        ax.set_xlabel(f"param for {x_method.name}", fontsize=26)
        ax.set_ylabel("mape", fontsize=26)

        plt.savefig(
            f"./main/results/extr_plots/{params['model']}_{params['f']}/{x_method.name}/{conf['max_cov']}_bw_{x_method.bw_list[n_bw]}.pdf"
        )
        plt.tight_layout()

    return True


def plot_extr_bw(conf, params, best_metrics_hyp, x_method):
    if len(x_method.bw_list) > 1:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(
            x_method.bw_list,
            best_metrics_hyp,
            label=f"{x_method.name}",
        )
        plt.legend(fontsize=26)
        plt.title(f"max_cov{conf['max_cov']}")
        ax.set_xlabel(f"bw", fontsize=26)
        ax.set_ylabel("best_mape for bw", fontsize=26)
        plt.savefig(
            f"./main/results/bw_plots/{params['model']}_{params['f']}/{x_method.name}/{conf['max_cov']}.pdf"
        )
        plt.tight_layout()

    return True
