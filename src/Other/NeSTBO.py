#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""

from optimization_loop import main
import torch

def fun(x): # sphere function
    return torch.sum(x**2, dim=1)  

    
dim = 10 # objective function dimensionality
Ninit = 50 # number of initial training data for GP
lb = -dim**2 # lower bound for decision variables
ub = dim**2 # upper bound for decision variables
delta = 0.5 # local searching space
M = dim # greedy inner selections per iteration
Budget = 300 # number of total budget
seed = 0

# torch.set_num_threads(4)
torch.manual_seed(seed)
params = torch.rand(1, dim).to(torch.float64) # starting point for NeST-BO
   
alg = main(fun, seed, dim, Ninit, lb, ub, params, Budget, delta, M)
X, Y, min_obj_list = alg.exec_alg()

