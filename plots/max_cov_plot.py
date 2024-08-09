import logging
import hydra
import matplotlib.pyplot as plt
from copy import copy
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from source.run import run
from source.methods import IS_method

log = logging.getLogger("__main__")
log.setLevel(logging.DEBUG)

@hydra.main(version_base=None, config_path="../config", config_name="config")
def max_cov_plot(cfg: DictConfig):
    
    #read configs:
    conf = OmegaConf.to_container(cfg["conf"])
    params = OmegaConf.to_container(cfg["params"])
    methods_names = OmegaConf.to_container(cfg["methods"])
    hyp_dict = OmegaConf.to_container(cfg["hyp_dict"])
    hyp_params_dict = OmegaConf.to_container(cfg["hyp_params_dict"])
    
    log.info(f"Starting configs:\n conf:\n{conf},\n params:\n{params},\n method_names:\n{methods_names},\n hyp_dict:\n{hyp_dict},\n hyp_params_dict:\n{hyp_params_dict}")

    bw_dict = {}
    #create additional methods_names if needed:
    if "KL" in hyp_params_dict["bw_list"]:
        hyp_params_dict["bw_list"].remove("KL")

        for s in copy(methods_names["x_estim"]):
            methods_names["x_estim"] += [s + '_KL']
            bw_dict[s] = copy(hyp_params_dict["bw_list"])
            
            bw_dict[s + '_KL'] = ['KL']

            if s in hyp_dict:
                hyp_dict[s + '_KL'] = hyp_dict[s]
            else:
                hyp_dict[s] = ['foo']
                hyp_dict[s + '_KL'] = ['foo']

        hyp_params_dict["bw_list"].append("KL")
    
    else:
        for s in methods_names["x_estim"]:
            bw_dict[s] = copy(hyp_params_dict["bw_list"])
            if s not in hyp_dict:
                hyp_dict[s] = ['foo']
    
    for s in methods_names["x_no_estim"] + methods_names["y"]:
        bw_dict[s] = ['g']
        if s not in hyp_dict:
            hyp_dict[s] = ['foo'] 
    
    methods_names_all = methods_names["x_estim"] + methods_names["x_no_estim"]
    print(f"Filled configs for methods:\n method_names:\n{methods_names_all},\n hyp_dict:\n{hyp_dict},\n bw_dict:\n{bw_dict}")
    
    #create method objects:
    methods_list = []
    for name in methods_names_all + methods_names["y"]:
        methods_list += [IS_method(name, params["n_tests"], hyp_dict[name], bw_dict[name])]
        
    for max_cov in tqdm(params["max_cov_list"]):
        conf["max_cov"] = max_cov
        run(conf, params, methods_list, hyp_params_dict)

    #PLOTS:
    methods_list_all = [method for method in methods_list if method.name != 'MCE_p']
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_all:
        ax.plot(params["max_cov_list"], x_method.best_metrics_dict["mape"], label=f"{x_method.name}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_all.pdf"
    )
    plt.tight_layout()

    methods_list_no_estim = [method for method in methods_list if method.name in ["MCE_g", "ISE", "ISE_clip"]]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_no_estim:
        ax.plot(params["max_cov_list"], x_method.best_metrics_dict["mape"], label=f"{x_method.name}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_no_estim.pdf"
    )
    plt.tight_layout()


    methods_list_estim = [method for method in methods_list if method.name in ["MCE_g", "ISE_deg", "ISE_uni", "ISE_estim", "ISE_estim_clip"]]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_estim:
        ax.plot(params["max_cov_list"], x_method.best_metrics_dict["mape"], label=f"{x_method.name}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_estim.pdf"
    )
    plt.tight_layout()

    methods_list_KL = [method for method in methods_list if method.name in ["MCE_g", "ISE_deg_KL", "ISE_uni_KL", "ISE_estim_KL", "ISE_estim_clip_KL"]]
    fig, ax = plt.subplots(figsize=(12, 12))
    for x_method in methods_list_KL:
        ax.plot(params["max_cov_list"], x_method.best_metrics_dict["mape"], label=f"{x_method.name}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("mape", fontsize=26)
    plt.savefig(
        f"./plots/results/max_cov_plots/{params['model']}_{params['f']}_max_cov_KL.pdf"
    )
    plt.tight_layout()

if __name__ == "__main__":
    max_cov_plot()
