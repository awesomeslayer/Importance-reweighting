import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from source.run import run
from source.simulation import visualize_GMM_config


@hydra.main(version_base=None, config_path='../config', config_name='config')
def max_cov_plot(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
 
    y = "MCE_p" 

    hyperparams_dict =  OmegaConf.to_container(cfg['hyperparams_dict'])
    params = OmegaConf.to_container(cfg['params'])
    hyperparams_params = OmegaConf.to_container(cfg['hyperparams_params'])
    conf = OmegaConf.to_container(cfg['conf'])
   
    errors_plot = {}
    for x in params['xs']:
        errors_plot[x] = []

    logging.debug(f"xs + y = {params['xs'] + [y]}")
    logging.debug(f"f = {params['f']}, model = {params['model']}")
    logging.debug(f"config: max_mu = {conf['max_mu']}, n_samples = {conf['n_samples']}, n_dim = {conf['n_dim']}, n_components = {conf['n_components']}, n_splits = {params['n_splits']}, kde_size = {hyperparams_dict['kde_size']}")
    logging.debug(f"regular_list = {hyperparams_dict['ISE_g_regular']}, clip_list = {hyperparams_dict['ISE_g_clip']},\n max_cov_list = {params['max_cov_list']}")
    logging.debug(f"n_tests = {params['n_tests']}")
    logging.debug(f"Status FindBestParam = {hyperparams_params['grid_flag']}")
    
    if(hyperparams_params['grid_flag']):
        n_hyp_tests = hyperparams_params['n_hyp_tests']
        logging.debug(f"n_hyp_tests = {n_hyp_tests}")

    for max_cov in params['max_cov_list']:
        log.info(f"max_cov:{max_cov}")
        conf["max_cov"] = max_cov

        error = {}
        for x in params['xs']:
            error[x] = 0

        elem = run(
            conf,
            params['f'],
            params['model'],
            params['n_splits'],
            params['xs'],
            y,
            n_tests=params['n_tests'],
            n_hyp_tests=n_hyp_tests,
            hyperparams_dict=hyperparams_dict,
            grid_flag=hyperparams_params['grid_flag'],
        )

        log.debug(f"errors for max_cov={max_cov}:\n {elem}")
        
        for x in params['xs']:
            error[x] += elem["mape"][x]
            errors_plot[x].append(error[x])

    fig, ax = plt.subplots(figsize=(12, 12))
    for x in params['xs']:
        ax.plot(params['max_cov_list'], errors_plot[x], label=f"{x}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    # ax.set_xscale("log")
    # plt.savefig("./plots/results/max_cov.pdf")
    plt.tight_layout()
    plt.show()

log = logging.getLogger(__name__)
if __name__ == "__main__":
    max_cov_plot()