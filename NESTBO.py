#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""

from optimization_loop import main
import torch
import matplotlib.pyplot as plt
import numpy as np

def fun(X): # Final Trid function value
    
    term1 = torch.sum((X - 1) ** 2, dim=1)  # Sum of squared terms
    term2 = torch.sum(X[:, 1:] * X[:, :-1], dim=1)  # Sum of product of consecutive terms
    
    return term1 - term2  

    
dim = 10
Ninit = 50 # number of initial training data for GP
lb = -dim**2
ub = dim**2
delta = 0.1
step = 0.1
epsilon_diff_acq_value = 0.1
max_samples_per_iteration = dim
bo_iter = 100
soln = -210 # true minimum
seed = 0

torch.set_num_threads(8)
torch.manual_seed(seed)
params = torch.rand(1, dim).to(torch.float64)
   
alg = main(fun, seed, dim, Ninit, lb, ub, params, bo_iter, delta, max_samples_per_iteration, epsilon_diff_acq_value, step)
X, Y, regret = alg.exec_alg()


plt.plot(np.array(regret) - soln)
plt.ylabel('Regret')
plt.xlabel('Iteration')

