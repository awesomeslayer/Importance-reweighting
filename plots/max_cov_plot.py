import numpy as np
import matplotlib.pyplot as plt
from source.run import run
import tqdm

if __name__ == "__main__":
    conf = dict()
    conf["max_mu"] = 50
    conf["n_samples"] = 4000 #10000
    conf["n_dim"] = 2
    conf["max_cov"] = 1
    conf["n_components"] = 15

    visualize_GMM_config(config = conf, alpha = 0.1)

    f = "linear"  # , ['GMM']
    model = "linear"  # ['boosting']

    n_splits = 1
    xs = [
        "MCE_g",
        "ISE_g",
        "ISE_g_regular",
        "ISE_g_estim",
        "ISE_g_clip",
        "ISE_g_estim_clip",
        "Mandoline",
    ]
    y = "MCE_p"

    errors_plot = {}
    for x in xs:
        errors_plot[x] = []

    hyperparams_dict = {
        "kde_size": [5],
        "ISE_g_regular": np.arange(0, 1, 0.025),
        "ISE_g_clip": np.arange(0, 1, 0.025),
        "ISE_g_estim_clip": np.arange(0, 1, 0.025),
    }


    max_cov_list = np.arange(1, 200, 10)
    n_tests = 30
    n_hyp_tests = 10

    for max_cov in tqdm(max_cov_list):
        conf["max_cov"] = max_cov

        error = {}
        for x in xs:
            error[x] = 0

        elem = run(
            conf,
            f,
            model,
            n_splits,
            xs,
            y,
            n_tests=n_tests,
            n_hyp_tests=n_hyp_tests,
            hyperparams_dict=hyperparams_dict,
            FindBestParam=True,
        )
        for x in xs:
            error[x] += elem["mape"][x]
            errors_plot[x].append(error[x])

    fig, ax = plt.subplots(figsize=(12, 12))
    for x in xs:
        ax.plot(max_cov_list, errors_plot[x], label=f"{x}")
    plt.legend(fontsize=26)
    ax.set_xlabel("max_cov", fontsize=26)
    ax.set_ylabel("errors", fontsize=26)
    #ax.set_xscale("log")
    #plt.savefig("./plots/results/max_cov.pdf")
    plt.tight_layout()
    plt.show()