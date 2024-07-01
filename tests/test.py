def test(f_gen, model, g_gen, p_gen, n_tests, n_splits = 1, target_error = None, 
         hyperparams = {'kde_size' : ['silverman'], 
                        'epsilon_reg' : [0],
                        'epsilon_clip' : [0]}):
    
    estimation_list = ['MCE_p', 'MCE_g', 'ISE_g', 'ISE_g_estim', 'ISE_g_regular', 
                        'ISE_g_clip', 'ISE_g_estim_clip', 
                        'MCE_p_variance', 'MCE_g_variance', 'ISE_g_variance', 'ISE_g_estim_variance', 'ISE_g_regular_variance']
    
    if target_error is None:
        target_error = estimation_list

    if isinstance(target_error, str):
        target_error = [target_error]

    for err in target_error:
        if err not in estimation_list:
            raise KeyError

    CV_err = dict()
    for err in target_error:
        CV_err[err] = []

    kf = KFold(n_splits=n_splits) \
        if n_splits > 1 else \
        ShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    for _ in trange(n_tests):
        iter_err = dict()
        for err in target_error:
            iter_err[err] = []

        f = f_gen()
        g_sample, g = g_gen()
        p_sample, p = p_gen()
        
        for i, (train_idx, test_idx) in enumerate(kf.split(g_sample)):
            
            best_error = {'ISE_g_regular' : 1e10,
                            'ISE_g_regular_variance' : 1e10,
                            'ISE_g_clip' : 1e10,
                            'ISE_g_estim_clip' : 1e10}
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
            best_epsilon = {'ISE_g_regular' : -1,
                             'ISE_g_regular_variance' : -1,
                             'ISE_g_clip' : -1,
                             'ISE_g_estim_clip' : -1}

            g_train = g_sample[train_idx]
            g_test = g_sample[test_idx]
            p_test = p_sample[test_idx]

            model.fit(g_train, f(g_train))
            err = lambda X: np.log(np.abs(f(X) - model.predict(X)))
            
            estim_list = ['ISE_g_estim', 'ISE_g_regular', 'ISE_g_estim_clip', 'ISE_g_estim_variance', 'ISE_g_regular_variance']
            if [i for i in target_error if i in estim_list]:
                kde = KernelDensity(kernel='gaussian', bandwidth=hyperparams['kde_size'][0]).fit(g_train)
                g_estim = lambda X: kde.score_samples(X)
                
                
            #our estimations:
            
            #may be add flag to print plots of params using counted errors for each param ?
            
            if 'MCE_p' in target_error:
                iter_err['MCE_p'] += [monte_carlo_error(err, p_test)]
                
            if 'MCE_g' in target_error:
                iter_err['MCE_g'] += [monte_carlo_error(err, g_test)]
                    
            if 'ISE_g' in target_error:
                iter_err['ISE_g'] += [importance_sampling_error(err, p, g, g_test)]
                
            if 'ISE_g_estim' in target_error:
                iter_err['ISE_g_estim'] += [importance_sampling_error(err, p, g_estim, g_test)]
                
            if 'ISE_g_regular' in target_error:
                errors_plot = []
                for i, epsilon_temp in enumerate(hyperparams['epsilon_reg']):
                    g_estim_new = lambda X: np.log((1 - epsilon_temp)*np.exp(g_estim(X)) + epsilon_temp/(conf['max_mu']**2)) 
                    
                    error = importance_sampling_error(err, p, g_estim_new, g_test)
                    errors_plot.append(error)
                    
                    if error < best_error['ISE_g_regular']:
                        best_error['ISE_g_regular'] = error
                        best_epsilon['ISE_g_regular'] = hyperparams['epsilon_reg'][i]
                
                fig, ax = plt.subplots(figsize = (12, 12))
                print(hyperparams['epsilon_reg'])
                print(errors_plot)
                ax.plot(hyperparams['epsilon_reg'], errors_plot, label = f'{x}')
                plt.legend(fontsize=26)
                ax.set_xlabel('epsilo', fontsize = 26)
                ax.set_ylabel('errors', fontsize = 26)
                ax.set_xscale('log')
                plt.tight_layout()
                plt.show() 

                iter_err['ISE_g_regular'] += [best_error['ISE_g_regular']]
                
            if 'ISE_g_clip' in target_error:
                for i in range(len(hyperparams['epsilon_clip'])):
                    error = ISE_clip(err, p, g, g_test, hyperparams['epsilon_clip'][i])
                    if error < best_error['ISE_g_clip']:
                        best_error['ISE_g_clip'] = error
                        best_epsilon['ISE_g_clip'] = hyperparams['epsilon_clip'][i]
                
                iter_err['ISE_g_clip'] += [best_error['ISE_g_clip']]
                
            if 'ISE_g_estim_clip' in target_error:
                for i in range(len(hyperparams['epsilon_clip'])):
                    error = ISE_clip(err, p, g_estim, g_test, hyperparams['epsilon_clip'][i])
                    if error < best_error['ISE_g_estim_clip']:
                        best_error['ISE_g_estim_clip'] = error
                        best_epsilon['ISE_g_estim_clip'] = hyperparams['epsilon_clip'][i]
                
                iter_err['ISE_g_estim_clip'] += [best_error['ISE_g_estim_clip']]
            
            #our variance estimations:
            if 'MCE_g_variance' in target_error:
                iter_err['MCE_g_variance'] += [monte_carlo_error_variance(err, g_test)]
            
            if 'ISE_g_variance' in target_error:
                iter_err['ISE_g_variance'] += [importance_sampling_error_variance(err, p, g, g_test)]
    
            if 'ISE_g_estim_variance' in target_error:
                iter_err['ISE_g_estim_variance'] += [importance_sampling_error_variance(err, p, g_estim, g_test)]
            
            if 'ISE_g_regular_variance' in target_error:
                for i, epsilon_temp in enumerate(hyperparams['epsilon_reg']):
                    g_estim_new = lambda X: np.log((1 - epsilon_temp)*np.exp(g_estim(X)) + epsilon_temp/(conf['max_mu']**2)) 
                    error = importance_sampling_error_variance(err, p, g_estim_new, g_test)
                    if error < best_error['ISE_g_regular_variance']:
                        best_error['ISE_g_regular_variance'] = error
                        best_epsilon['ISE_g_regular_variance'] = hyperparams['epsilon_reg'][i]
                
                iter_err['ISE_g_regular_variance'] += [best_error['ISE_g_regular_variance']]
            
            #if 'ISE_g_clip_variance' in target_error:
                
            #if 'ISE_g_estim_clip_variance' in target_error
                       
        for err in target_error:
            CV_err[err] += [logsumexp(iter_err[err]) - np.log(n_splits)]

    return CV_err
