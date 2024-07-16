import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from source.run import run
from source.simulation import visualize_GMM_config


@hydra.main(version_base=None, config_path="../config", config_name="config")
def max_cov_plot(cfg: DictConfig):
    hyperparams_dict = OmegaConf.to_container(cfg["hyperparams_dict"])
    params = OmegaConf.to_container(cfg["params"])
    hyperparams_params = OmegaConf.to_container(cfg["hyperparams_params"])
    conf = OmegaConf.to_container(cfg["conf"])

    errors_plot = {}
    for x in params["x"] + params["x_hyp"]:
        errors_plot[x] = []

    logging.info(f"\nxs + y = {params['x'] + params['x_hyp'] + params['y']}\n")
    logging.info(f"\nf = {params['f']}, model = {params['model']}\n")
    logging.info(
        f"\nconfig: max_mu = {conf['max_mu']}, n_samples = {conf['n_samples']}, n_dim = {conf['n_dim']}, n_components = {conf['n_components']}, n_splits = {params['n_splits']}, kde_size = {hyperparams_dict['kde_size']}\n"
    )
    logging.info(
        f"\nslices_list = {hyperparams_dict['Mandoline']},\n regular_list = {hyperparams_dict['ISE_g_regular']},\n clip_list = {hyperparams_dict['ISE_g_clip']},\n max_cov_list = {params['max_cov_list']}\n"
    )
    logging.info(f"\nn_tests = {params['n_tests']}\n")
    logging.info(
        f"\ngrid_flag = {hyperparams_params['grid_flag']}, smootg_clipping = {hyperparams_params['smooth_flag']}\n"
    )

    for max_cov in params["max_cov_list"]:
        log.info(f"max_cov:{max_cov}")
        conf["max_cov"] = max_cov

        error = {}
        for x in params["x"] + params["x_hyp"]:
            error[x] = 0

        elem = run(
            conf,
            params,
            hyperparams_params,
            hyperparams_dict,
        )

        logging.info(f"\nerrors for max_cov={max_cov}:\n {elem}\n")

        for x in params["x"] + params["x_hyp"]:
            errors_plot[x] += elem["mape"][x]
            
    fig, ax = plt.subplots(figsize=(12, 12))
    for x in params["x"] + params["x_hyp"]:
        ax.plot(params["max_cov_list"], errors_plot[x], label=f"{x}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig("./plots/results/max_cov.pdf")
    plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(12, 12))
    for x in ["MCE_g", "ISE_g_estim", "ISE_g_regular", "ISE_g_estim_clip", "Mandoline"]:
        ax.plot(params["max_cov_list"], errors_plot[x], label=f"{x}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig("./plots/results/max_cov_estim.pdf")
    plt.tight_layout()
    # plt.show()

    fig, ax = plt.subplots(figsize=(12, 12))
    for x in ["MCE_g", "ISE_g", "ISE_g_clip", "Mandoline"]:
        ax.plot(params["max_cov_list"], errors_plot[x], label=f"{x}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig("./plots/results/max_cov_no_estim.pdf")
    plt.tight_layout()
    # plt.show()


log = logging.getLogger(__name__)
if __name__ == "__main__":
    max_cov_plot()
