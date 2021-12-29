
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import numpy as np
import emcee
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import integrate



X = [0.0, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2, 0.23, 0.25, 0.28, 0.3, 0.33, 0.35, 0.38, 0.4, 0.42, 0.45, 0.47, 0.5, 0.53, 0.55, 0.57, 0.6, 0.62, 0.65, 0.68, 0.7, 0.72, 0.75, 0.78, 0.8, 0.82, 0.85, 0.88, 0.9, 0.93, 0.95, 0.97]
Y = [3.15, 3.01, 3.29, 3.61, 3.13, 3.18, 3.77, 3.58, 3.26, 3.61, 3.36, 3.41, 3.67, 3.08, 3.18, 3.58, 3.5, 3.94, 3.63, 3.53, 4.44, 3.98, 4.12, 3.72, 4.04, 4.28, 3.95, 4.46, 4.22, 4.36, 4.32, 5.11, 4.6, 4.33, 4.95, 4.38, 4.86, 4.26, 4.5, 5.01]
X=np.array(X)
Y=np.array(Y)

# Data Exploration: 

slope, intercept, r_value, p_value, slope_std_error = stats.linregress(X,Y)
predict_y = slope * X + intercept

sns.set(rc={'figure.figsize':(8, 8)})
ax = sns.regplot(x=X, y=Y, line_kws={'label':'$y=%3.4s*x+%3.4s$'%(slope, intercept)});
sns.regplot(x=X, y=Y, fit_reg=False, ax=ax);
sns.regplot(x=X, y=predict_y,scatter=False, ax=ax);
ax.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('OLS Fit')
plt.show()

# 12.3.1 - two different approaches: 
if True:
	ts=1.96
	# Method 1: Scipy stats package
	result = stats.linregress(X,Y)
	slope,intercept = result.slope,result.intercept
	slope_95,intercept_95 = ts*result.stderr,ts*result.intercept_stderr

	m='Slope = {:.2f} 95% CI: [{:.2f},{:.2f}]'.format(slope,slope-slope_95,slope+slope_95)
	b ='Intercept = {:.2f} 95% CI: [{:.2f},{:.2f}]'.format(intercept,intercept-intercept_95,intercept+intercept_95)
	print(m,'\n',b)


	# Method 2: Statsmodels 
	x=sm.add_constant(X)
	mod = sm.OLS(Y, x)
	res = mod.fit()
	print(res.summary())


#12.3.2
# Reference: 'https://statswithr.github.io/book/introduction-to-bayesian-regression.html'


sigma = 0.3


def gaussian(x,mu,s=sigma):
    v = (1/(s*np.sqrt((2*np.pi))))*np.exp(-0.5*((x-mu)/s)**2)
    return v

# a,b will be constrained to {-5,5} anyway
def prior(c):
    if abs(c) > 5:
        return 0
    return 1

a = np.linspace(-5,5,1000)
b = np.linspace(-5,5,1000)

def likelihood(a,b):
    l=1
    for i in range(len(X)):
        l*=gaussian(Y[i],(b+a*X[i]))
    return l

# Most readings ignore the evidence since it's a constant
# and often times not easy to compute. It makes sense since
# all we need is the propability maximum/peak.
#'https://www.countbayesie.com/blog/2019/6/12/logistic-regression-from-bayes-theorem'

def evidence():

    evd = lambda b, a: np.product([gaussian(Y[i],(b+a*X[i])) for i in range(len(X))])
    # using scipy.integrate.dblquad() method

    P_D = integrate.dblquad(evd, -5,5 , lambda a: -5, lambda a: 5)
    return P_D[0]


P_D=evidence()

matrix = []

for i in range(len(a)):
    row=[]
    for j in range(len(b)):
        Posterior= likelihood(a[i],b[j])/P_D
        row.append(Posterior)
    matrix.append(row)
    row=[]

m = plt.pcolor(a, b, matrix,cmap=plt.get_cmap('viridis', 20))
cb = plt.colorbar(m)
plt.ylabel('Intercept: b')
plt.xlabel('Slope: a')
cb.ax.set_title(r'$p((a,b)|\mathcal{D})$')
plt.xlim([2, 4])
plt.ylim([1,3])
plt.title('12.3.2: P((a,b)|D) Heatmap')
plt.show()



# 3-4: Plot Posteriors and 95% CIs:
# Reference: 'https://www.programcreek.com/python/example/103021/emcee.EnsembleSampler'

# Log works best with emcee 

def log_likelihood(xs, data):
    a,b = xs
    x, y = data
    reg = (a*x + b)-y
    
    return stats.norm.logpdf(reg).sum()

ndim = 2  # num of params
nwalk = 50  # num of walks
p0 = np.random.uniform(low=-5, high=5, size=(nwalk, ndim))  # initial position
sampler = emcee.EnsembleSampler(nwalk, ndim, log_likelihood, args=[(X, Y)])
state = sampler.run_mcmc(p0, 1000)  
chain = sampler.chain[:, 100:, :] # discard first 100
flat_chain = chain.reshape((-1, ndim))  

samples = sampler.get_chain()

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

labels = ["m", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")


import corner

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
fig = corner.corner(
    flat_samples, labels=labels,plot_density=True,fill_contours=True,color='green',quantiles=[0.05,0.5,0.95],show_titles=True
);


for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [5, 50, 95])
    q = np.diff(mcmc)
    param = "{3} = {0:.3f} 95% CI [{1:.3f},{2:.3f}]"
    param = param.format(mcmc[1], mcmc[1]-q[0],mcmc[1]+ q[1], labels[i])
    print(param)

plt.show()

