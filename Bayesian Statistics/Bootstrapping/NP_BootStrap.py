import random,math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

measurements=np.array([4.72, 9.11, 5.57, 7.66, 4.69, 4.86, 6.51, 6.65, 4.77, 0.26, 4.67, 6.37, 5.05, 5.91, 5.54, 2.13, 6.77, 3.84, 4.0, 6.18, 3.54, 5.52, 3.29, 4.62, 4.25, 4.08, 3.37, 4.91, 5.24, 6.85, 3.85, 5.11, 9.41, 5.78, 5.97, 5.87, 1.59, 4.51, 0.72, 6.72, 8.4, 3.94, 8.53, 2.76, 2.62, 6.11, 3.37, 4.01, 7.17, 3.05])
N=10000
def bootstrap(Y,size,mode):
    
    Ystar = np.empty(size)
    
    # Bootstraping 
    for i in range(size):
        samples = np.random.choice(Y,size=len(Y))
        if mode == 'mean':
            Ystar[i] = np.mean(samples)
        if mode == 'std':
            Ystar[i] = np.std(samples,ddof=1)

    return Ystar


if True: 
	bootstrap_sample = bootstrap(measurements,N,'mean')

	CI_95 = np.percentile(bootstrap_sample,[2.5,97.5])


	bins = np.linspace(4, 6, 11)
	# Bootstrap PDF Plot
	plt.hist(bootstrap_sample,density=True,color='green',bins=np.arange(min(measurements), max(measurements) + .1, .1))

	# 95% confidence interval
	plt.axvline(x=np.percentile(bootstrap_sample,[2.5]), ymin=0, ymax=1,label='2.5th percentile',color='black',linestyle='dashed')
	plt.axvline(x=np.percentile(bootstrap_sample,[97.5]), ymin=0, ymax=1,label='97.5th percentile',color='red',linestyle='dashed')
	plt.axvline(x=np.percentile(bootstrap_sample,[50]), ymin=0, ymax=1,label='mean',color='yellow',linestyle='dashed')

	plt.xlabel("$\overline {X}$")
	plt.ylabel("Frequency")
	plt.title(r'Bootstrap : $\overline {X}$'+"= {0:.3f} ; 95 % C.I = [{1:.3f}, {2:.3f}]".format(np.mean(bootstrap_sample),CI_95[0],CI_95[1]))
	plt.legend()
	plt.grid()
	plt.xlim([3.8, 6.2])
	plt.show()

if True:
	bootstrap_sample = bootstrap(measurements,N,'std')

	CI_95 = np.percentile(bootstrap_sample,[2.5,97.5])

	bins = np.linspace(4, 6, 11)
	# Bootstrap PDF Plot
	plt.hist(bootstrap_sample,density=True,color='green',bins=np.arange(min(measurements), max(measurements) + .1, .1))

	# 95% confidence interval
	plt.axvline(x=np.percentile(bootstrap_sample,[2.5]), ymin=0, ymax=1,label='2.5th percentile',color='black',linestyle='dashed')
	plt.axvline(x=np.percentile(bootstrap_sample,[97.5]), ymin=0, ymax=1,label='97.5th percentile',color='red',linestyle='dashed')
	plt.axvline(x=np.percentile(bootstrap_sample,[50]), ymin=0, ymax=1,label='std',color='yellow',linestyle='dashed')

	plt.xlabel("$\overline {\sigma}$")
	plt.ylabel("Frequency")
	plt.title(r'Bootstrap : $\overline {\sigma}$'+"= {0:.3f} ; 95 % C.I = [{1:.3f}, {2:.3f}]".format(np.mean(bootstrap_sample),CI_95[0],CI_95[1]))
	plt.legend()
	plt.grid()
	plt.xlim([1, 2.7])
	plt.show()



if True:
	# Bayesian 1:  
	mean, var, std = stats.bayes_mvs(measurements)
	print(mean,'\n',std)


	# Bayesian Model 2:

	import pymc3 as pm

	model = pm.Model()

	if __name__ == '__main__':

	    with model:
		    
		    # Prior
	        mu = pm.Normal('mu',4)
	        sigma = pm.Normal('sigma', 1.5)
		    
		    # Likelihood function.
	        y_obs = pm.Normal('y_obs', mu=mu,sd=sigma, observed=measurements)
		    
		    # Try 330 draws and 3 chains.
	        trace = pm.sample(draws=330, chains=3)


	    pm.traceplot(trace)
	    pm.plot_posterior(trace)


	plt.show()
