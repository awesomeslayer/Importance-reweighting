import numpy as np

from source.run import run

conf = dict()
conf["max_mu"] = 100
conf["n_samples"] = 50000
conf["n_dim"] = 2
conf["max_cov"] = 5
conf["n_components"] = 30
n_splits = 1


f = "linear"
model = "linear"

xs = ["ISE_g_clip", "ISE_g_estim_clip"]
y = "MCE_p"

hyperparams = {
    "kde_size": [1],
    "epsilon_clip": np.arange(0, 1, 0.001),
}

run(conf, f, model, n_splits, xs, y, n_tests=1, hyperparams=hyperparams)
