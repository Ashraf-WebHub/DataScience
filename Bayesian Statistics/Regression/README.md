## The Markov Chain Monte Carlo Methods for Linear Regression

### Why MCMC?

MCMC methods are used when integration of the evidence in Bayes Theorem becomes analytically impossible, and numerical integration grows computationally expensive with higher dimensions (multiple parameters).
MCMC methods directly sample the posterior distribution and use Markov Chains for convergence criteria. 

![image](https://user-images.githubusercontent.com/78180239/147704707-24b6eaaf-041f-45fc-9509-409e55c58574.png)

Reference: Kruschke, J. K. (2011). Doing Bayesian data analysis: A tutorial with R and BUGS. Elsevier Academic Press.



## Ordinary Least Squares Technique:

<img src="https://user-images.githubusercontent.com/78180239/148709116-b23cd769-cca6-4471-9cc0-5fbf2aeeda79.png" width="60%" height="60%">


## Compare Frequentist and Bayesian Libraries:

<img src="https://user-images.githubusercontent.com/78180239/147704741-723d7379-bab1-4a78-8bb8-7114f23583ec.png" width="60%" height="60%">

## Markov Chain Monte Carlo Technique:

### Manual implementation:

<img src="https://user-images.githubusercontent.com/78180239/147704752-dbb27eae-1475-46d7-b7b4-276b11c24878.png" width="60%" height="60%">

### Validate against emcee library:

<img src="https://user-images.githubusercontent.com/78180239/147704771-a4140bcf-908b-4a79-8231-146aa4011e02.png" width="60%" height="60%">
Converging Markov Chains for Regression parameters slope 'm' & intercept 'b':
<img src="https://user-images.githubusercontent.com/78180239/147704783-32d34d35-5843-4dc4-a692-2b52f89f4b35.png" width="60%" height="60%">

