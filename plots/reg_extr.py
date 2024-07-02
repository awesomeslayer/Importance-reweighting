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

xs = ["ISE_g_regular"]
y = "MCE_p"

hyperparams = {
    "kde_size": [1],
    "epsilon_reg": [
        0,
        0.0000000000000000000001,
        0.000000000000001,
        0.00000000000001,
        0.0000000000001,
        0.00000000001,
        0.000000001,
        0.00000001,
        0.0000001,
        0.0001,
        0.0005,
        0.001,
        0.01,
        0.05,
        0.1,
        0.15,
        0.3,
        0.5,
        1,
        1.5,
        2,
        10,
    ],
}

run(conf, f, model, n_splits, xs, y, n_tests=1, hyperparams=hyperparams)
