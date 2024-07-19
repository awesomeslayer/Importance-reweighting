import numpy as np
import statsmodels.api as sm #for library lscv, beta = 0
from scipy.integrate import dblquad #for integrating (longer)
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity

def monte_carlo_integration(f, num_samples=10000):
    x_samples = np.random.normal(0, 1, num_samples)
    y_samples = np.random.normal(0, 1, num_samples)
    
    evaluations = np.array([f(x, y) for x, y in zip(x_samples, y_samples)])
    integral_estimate = np.mean(evaluations)
    
    return integral_estimate

def squared_error_sklearn(h, sample):
    h = h[0]
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(sample)
    f_n_square = lambda x, y: (np.exp(kde.score_samples(np.array([[x, y]]))))**2
                
    # Compute the mean of cross prediction value
    def f_sub_sample_mean():
        summation = 0
        for i in range(len(sample)):
            subsample = np.delete(sample, i, axis=0)
            
            kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(subsample)
            predict_value = np.exp(kde.score_samples(sample[i].reshape(1, -1)))
            summation += predict_value
        return summation / len(sample)
    
    integral = monte_carlo_integration(f_n_square)
    #print(f"integral : {integral}")
    cv = abs(integral - 2 * f_sub_sample_mean()) 
    print(f"cv : {cv}")
    return cv

# Find the optimal bandwidth h minimizing cv and return it
def KL_LSCV_find_bw(sample, beta = 0):
    h0 = np.array(sample).std() * (len(sample) ** (-0.2))
    
    #print(f"Initial h0: {h0}")
    
    # Constraint to ensure h is larger than 10**(-8)
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 10**(-8)})
    
    res = minimize(squared_error_sklearn, h0, args=(sample), constraints=cons)
    h = res.x # The optimal h
        
    return h[0]