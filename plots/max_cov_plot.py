import numpy as np
import matplotlib.pyplot as plt
from source.run import run

conf = dict()
conf["max_mu"] = 100
conf["n_samples"] = 50000
conf["n_dim"] = 2
conf["max_cov"] = 100
conf["n_components"] = 30

f = "linear"  # , ['GMM']
model = "linear"  # ['boosting']

n_splits = 1
xs = [
     "MCE_g",
     "ISE_g",
    # "ISE_g_regular",
    # "ISE_g_estim",
    # "ISE_g_clip",
    # "ISE_g_estim_clip",
    "Mandoline",
]
y = "MCE_p"

errors_plot = {}
for x in xs:
    errors_plot[x] = []

hyperparams_dict = {
    "kde_size": ["scott"],
    "ISE_g_regular": np.arange(0.1, 1, 0.6),
    "ISE_g_clip": np.arange(0.1, 1, 0.6),
    "ISE_g_estim_clip": np.arange(0.1, 1, 0.6),
}


max_cov_list = np.arange(200, 1, -20)
n_tests = 1
n_hyp_tests = 1

for max_cov in max_cov_list:
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
        FindBestParam=False,
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
ax.set_xscale("log")
plt.savefig("./plots/results/max_cov.pdf")
plt.tight_layout()
# plt.show()
