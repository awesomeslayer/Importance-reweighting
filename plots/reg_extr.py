#ise_regular(epsilon) - plot for optimum + check limits, + variance plot :

fs = ['linear']#, 'GMM']
models = ['linear']# + ['boosting']
n_splits = 1
xs = ['ISE_g_regular', 'ISE_g_estim']#, 'ISE_g_regular_variance'] 
y = 'MCE_p'

errors_plot = {}
for x in xs:
    errors_plot[x] = []

epsilons_reg = [0]

hyperparams = {'kde_size' : ['scott'],
               'epsilon_reg' : [0],
               'epsilon_clip' : [0]}

n_tests = 1
for f in fs:
    for model in models:
        for epsilon in epsilons_reg:
            
            error = {}
            for x in xs:
                error[x] = 0
                
            hyperparams['epsilon_reg'] = [epsilon]   
            
            for n in range(n_tests):
                elem = run(conf, f, model, n_splits, xs, y, hyperparams = hyperparams)
                
                for x in xs:
                    error[x] += elem['mape'][x]/n_tests
                
            for x in xs:
                errors_plot[x].append(error[x])

for i, epsilon in enumerate(epsilons_reg):
    delta = errors_plot['ISE_g_estim'][i] - errors_plot['ISE_g_regular'][i]
    print(f'{epsilon} delta: {delta}',)
                
#errors                
fig, ax = plt.subplots(figsize = (12, 12))
for x in xs:
    ax.plot(epsilons_reg, errors_plot[x], label = f'{x}')
plt.legend(fontsize=26)
ax.set_xlabel('epsilon_reg', fontsize=26)
ax.set_ylabel('error', fontsize=26)
plt.tight_layout()
plt.show() 