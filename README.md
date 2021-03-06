# Variational Sequential Monte Carlo

The repository contains code for the variational sequential Monte Carlo (VSMC) algorithm for approximate Bayesian inference:
```
Variational Sequential Monte Carlo.
Christian A. Naesseth, Scott W. Linderman, Rajesh Ranganath, and David M. Blei
Proceedings of the 21st International Conference on Artificial Intelligence and Statistics 2018,
Lanzarote, Spain.
```
Furthermore, it contains a simulation example (a linear Gaussian state space model) on how to use the VSMC module. 
Note that this example learns both model parameters and proposal parameters so the final lower bound will not be a lower 
bound to the exact log-marginal likelihood for the parameters that generated the data.
