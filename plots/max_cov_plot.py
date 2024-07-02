import numpy as np
import matplotlib as plt

from source.run import run

conf = dict()
conf['max_mu'] = 100
conf['n_samples'] = 50000
conf['n_dim'] = 2
conf['max_cov'] = 100
conf['n_components'] = 30

fs = ['linear']#, 'GMM']
models = ['linear']# + ['boosting']

n_splits = 1
xs = ['MCE_g', 'ISE_g', 'ISE_g_regular', 'ISE_g_estim']
y = 'MCE_p'

errors_plot = {}
for x in xs:
    errors_plot[x] = []

hyperparams = {'kde_size' : ['silverman'],
               'epsilon_reg' : np.arange(0, 1, 0.1),
               'epsilon_clip' : np.arange(0.1, 0.9, 0.1)}


max_cov_list = [1, 3, 5, 10, 25, 50, 70, 100]

for f in fs:
    for model in models:
        for max_cov in max_cov_list:
            conf['max_cov'] = max_cov
            
            error = {}
            for x in xs:
                error[x] = 0
        
            elem = run(conf, f, model, n_splits, xs, y, n_tests = 10, hyperparams = hyperparams)
            for x in xs:
                error[x] += elem['mape'][x]
                errors_plot[x].append(error[x])                

fig, ax = plt.subplots(figsize = (12, 12))
for x in xs:
    ax.plot(max_cov_list, errors_plot[x], label = f'{x}')
plt.legend(fontsize=26)
ax.set_xlabel('max_cov', fontsize = 26)
ax.set_ylabel('errors', fontsize = 26)
plt.savefig("../plots/results/max_cov.pdf")
plt.tight_layout()
plt.show() 