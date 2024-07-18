import matplotlib.pyplot as plt

def extr_plots(conf, params, metrics_hyp_dict, hyperparams_dict, log_flag = False):
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
        if log_flag:
            ax.set_xscale('log')
        plt.savefig(
            f"./plots/results/{params['model']}_{params['f']}_{x_temp}_{conf['max_cov']}.pdf"
        )
        plt.tight_layout()
        #plt.show()
