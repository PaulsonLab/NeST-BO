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

# def fun(x):
#     term1 = (10*x[:,0]**2 + 10*x[:,1]**2)/2
#     term2 = 5*torch.log(1 + torch.exp(-x[:, 0] - x[:,1]))
    
#     return term1 + term2

# def fun(x):
    
    
#     return 10*x[:, 1]**2 + x[:,0]**2


    
dim = 2
Ninit = 8 # number of initial training data for GP
lb = -4
ub = 4
delta = 0.2
step = 0.1
epsilon_diff_acq_value = 0.1
max_samples_per_iteration = 4
bo_iter = max_samples_per_iteration+1
soln = -210 # true minimum
seed = 0

torch.set_num_threads(8)
torch.manual_seed(seed)
# params = torch.rand(1, dim).to(torch.float64)
params = torch.tensor([[0.25, 0.25]]).to(torch.float64)
   
alg = main(fun, seed, dim, Ninit, lb, ub, params, bo_iter, delta, max_samples_per_iteration, epsilon_diff_acq_value, step)
X, Y, regret = alg.exec_alg()


# X = torch.load('NeSTBO.pt')
# X = torch.load('GIBO.pt')

# Define 2D grid
x = np.linspace(lb, ub, 400)
y = np.linspace(lb, ub, 400)
X1, X2 = np.meshgrid(x, y)

# Flatten and stack as input to the function
X_grid = np.stack([X1.ravel(), X2.ravel()], axis=1)
X_tensor = torch.tensor(X_grid, dtype=torch.float32)

# Evaluate the function
Z = fun(X_tensor).detach().numpy().reshape(X1.shape)

# Plotting
plt.figure(figsize=(8, 6))
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)

plt.scatter(lb+(ub-lb)*X[max_samples_per_iteration+2:,0], lb+(ub-lb)*X[max_samples_per_iteration+2:,1], label='Training data', color='black', s = 200, marker='x')
plt.scatter(lb+(ub-lb)*params[0,0], lb+(ub-lb)*params[0,1], label='Starting  Location', color = 'red', marker = '*', s = 500)
plt.scatter(lb+(ub-lb)*X[1:max_samples_per_iteration+1,0], lb+(ub-lb)*X[1:max_samples_per_iteration+1,1], label='Sample points (NeST-BO)', color='purple', s = 200)
plt.scatter(lb+(ub-lb)*X[0,0], lb+(ub-lb)*X[0,1], label='Newton direction', color='orange', marker = '*', s = 500)

plt.title("2D Contour Plot of Trid Function")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()




