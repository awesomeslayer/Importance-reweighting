import numpy as np
import matplotlib as plt

from source.run import run

conf = dict()
conf['max_mu'] = 100
conf['n_samples'] = 50000
conf['n_dim'] = 2
conf['max_cov'] = 100
conf['n_components'] = 30

#clip(epsilon) -plot for optimum + check limits:
fs = ['linear']#, 'GMM']
models = ['linear']# + ['boosting']

n_splits = 1
xs = ['ISE_g_clip', 'ISE_g'] 
y = 'MCE_p'

errors_plot = {}
for x in xs:
    errors_plot[x] = []

epsilons_clip = [0.1, 0.3, 0.7, 10, 100, 10000]

hyperparams = {'kde_size' : ['scott'],
               'epsilon_reg' : [0],
               'epsilon_clip' : [0]}

n_tests = 1
for f in fs:
    for model in models:
        for epsilon_clip in epsilons_clip:
            
            error = {}
            for x in xs:
                error[x] = 0
        
            hyperparams['epsilon_clip'] = [epsilon_clip]   
            
            for n in range(n_tests):
                elem = run(conf, f, model, n_splits, xs, y, hyperparams = hyperparams)
                for x in xs:
                    error[x] += elem['mape'][x]/n_tests
                    
            for x in xs:
                errors_plot[x].append(error[x])
                
fig, ax = plt.subplots(figsize = (12, 12))
for x in xs:
    ax.plot(epsilons_clip, errors_plot[x], label = f'{x}')

plt.legend(fontsize = 26)
ax.set_xlabel('epsilon_clip', fontsize = 26)
ax.set_ylabel('error', fontsize = 26)
plt.savefig("results/clip.pdf")
plt.tight_layout()
plt.show() 