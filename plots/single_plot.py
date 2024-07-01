from source.run import run
import numpy as np

conf = dict()
conf['max_mu'] = 100
conf['n_samples'] = 50000
conf['n_dim'] = 2
conf['max_cov'] = 100
conf['n_components'] = 30

f = ['linear']#, 'GMM']
model = ['linear'] #+ ['boosting']
n_splits = 1
xs = ['ISE_g_regular', 'ISE_g_estim']
y = 'MCE_p'

hyperparams = {'kde_size' : [1],
              'epsilon_reg' : np.arange(0, 1, 0.01),
              'epsilon_clip' : [0.2]}

elem = run(conf, f, model, n_splits, xs, y, hyperparams = hyperparams)
print(f"error of reg: {elem['mape']['ISE_g_regular']}")
print(f"error of IS: {elem['mape']['ISE_g_estim']}")

