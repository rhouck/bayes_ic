import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.distributions.timeseries import *
from theano.tensor import repeat

class BayesIC(object):
    
    def __init__(self, signal, target):
        
        for i in (signal, target):
            if not isinstance(i, pd.DataFrame):
                raise TypeError('signal and target must be pandas dataframes')
        cols = sorted(set(signal.columns) & set(target.columns))
        self.signal = signal[cols]
        self.target = target[cols]
      
    def get_stats(self, samples):
        s = pd.Series(np.sort(samples))
        med = s.median()
        avg = s.mean()
        prob_sub_zero = s.map(lambda x: 1 if x < 0 else 0).sum() / (s.shape[0] * 1.)
        return {'median': med,
                'mean': avg,
                'prob_sub_zero': prob_sub_zero,
                'std': s.std()}
    
    def plot_trace(self, trace):
        pm.traceplot(trace)
        
    def fit_all(self, samples=500):
        betas = []
        for i in self.target.columns:
            print
            print i
            stats, trace = self._fit_model(self.signal[i], self.target[i], samples)
            stats['Id'] = i 
            betas.append(stats)
        self.betas = pd.DataFrame(betas)
    
    def mod_signal(self):
        
        try:
            self.betas
        except:
            raise Exception('must run `fit all` before `mod_signal`')
        
        return self.signal * self.betas.set_index('Id')['prob_sub_zero']
        
    
    def _fit_model(self, signal, target, samples):

        with pm.Model() as model:

            # define priors
            alpha = pm.Normal('alpha', mu=0, sd=20)
            beta = pm.Normal('beta', mu=0, sd=20)
            sigma = pm.Uniform('sigma', lower=0, upper=20)

            # define linear regression
            y_est = alpha + beta * signal.values

            # define likelihood
            likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=target.values)

            # inference
            start = pm.find_MAP() # Find starting value by optimization
            step = pm.NUTS(state=start) # Instantiate MCMC sampling algorithm
            trace = pm.sample(samples, step, progressbar=True)
        
        return self.get_stats(trace.beta), trace