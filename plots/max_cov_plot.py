import logging

import hydra
import matplotlib.pyplot as plt
import tqdm
from omegaconf import DictConfig, OmegaConf

from source.run import run


@hydra.main(version_base=None, config_path="../config", config_name="config")
def max_cov_plot(cfg: DictConfig):
    hyperparams_dict = OmegaConf.to_container(cfg["hyperparams_dict"])
    params = OmegaConf.to_container(cfg["params"])
    hyperparams_params = OmegaConf.to_container(cfg["hyperparams_params"])
    conf = OmegaConf.to_container(cfg["conf"])

    methods_list = params["x"] + params["x_hyp"]
    if "KL_LSCV" in hyperparams_params["bw_list"]:
        methods_list += ["ISE_g_estim_KL", "ISE_g_regular_KL", "ISE_g_estim_clip_KL"]

    errors_plot = {}
    for x in methods_list:
        errors_plot[x] = []

    log.info(f"\nxs + y = {params['x'] + params['x_hyp'] + params['y']}\n")
    log.info(f"\nf = {params['f']}, model = {params['model']}\n")
    log.info(
        f"\nconfig: max_mu = {conf['max_mu']}, n_samples = {conf['n_samples']}, n_dim = {conf['n_dim']}, n_components = {conf['n_components']}, n_splits = {params['n_splits']}, log_flag = {params['log_flag']}\n"
    )
    log.info(
        f"\nslices_list = {hyperparams_dict['Mandoline']},\n regular_list = {hyperparams_dict['ISE_g_regular']},\n clip_list = {hyperparams_dict['ISE_g_clip']},\n max_cov_list = {params['max_cov_list']}\n"
    )
    log.info(f"\nn_tests = {params['n_tests']}\n")
    log.info(
        f"\nbw_list = {hyperparams_params['bw_list']},grid_flag =  {hyperparams_params['grid_flag']}, smooth_clipping = {hyperparams_params['smooth_flag']}, slice_method = {hyperparams_params['slice_method']}\n"
    )

    for max_cov in tqdm.tqdm(params["max_cov_list"]):
        log.info(f"max_cov:{max_cov}")
        conf["max_cov"] = max_cov

        elem = run(conf, params, hyperparams_params, hyperparams_dict, log)

        log.info(f"\nmape's and rmse's for max_cov={max_cov}:\n {elem}\n")

        for x_temp in methods_list:
            errors_plot[x_temp] += [elem["mape"][x_temp]]

    fig, ax = plt.subplots(figsize=(12, 12))
    for x_temp in methods_list:
        ax.plot(params["max_cov_list"], errors_plot[x_temp], label=f"{x_temp}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov.pdf"
    )
    plt.tight_layout()
    # plt.show()


log = logging.getLogger(__name__)
if __name__ == "__main__":
    max_cov_plot()
