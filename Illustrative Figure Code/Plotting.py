#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 13:51:11 2025

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


dim = 2
Ninit = 8 # number of initial training data for GP
lb = -4
ub = 4
delta = 1
step = 0.1
epsilon_diff_acq_value = 0.1
max_samples_per_iteration = 4
bo_iter = max_samples_per_iteration+1
soln = -210 # true minimum
seed = 0

params = torch.tensor([[0.25, 0.25]]).to(torch.float64)

X = torch.load('NESTBO.pt')
X_gibo = torch.load('GIBO.pt')

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
plt.figure(figsize=(12, 12),dpi=150)
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)

plt.scatter(lb+(ub-lb)*X[max_samples_per_iteration+2:,0], lb+(ub-lb)*X[max_samples_per_iteration+2:,1], label='Training data', color='black', s = 100, marker='x')
plt.scatter(lb+(ub-lb)*params[0,0], lb+(ub-lb)*params[0,1], label='Starting  Location', color = 'red', marker = '*', s = 500)
plt.scatter(lb+(ub-lb)*X[1:max_samples_per_iteration+1,0], lb+(ub-lb)*X[1:max_samples_per_iteration+1,1], label='Sample points (NeST-BO)', color='orange', s = 200)
plt.scatter(lb+(ub-lb)*X[0,0], lb+(ub-lb)*X[0,1], label='Newton direction(NeST-BO)', color='orange', marker = '*', s = 500)


plt.scatter(lb+(ub-lb)*X_gibo[1:max_samples_per_iteration+1,0], lb+(ub-lb)*X_gibo[1:max_samples_per_iteration+1,1], label='Sample points (GIBO)', color='lightgreen', s = 200)
plt.scatter(lb+(ub-lb)*X_gibo[0,0], lb+(ub-lb)*X_gibo[0,1], label='Newton direction (GIBO)', color='lightgreen', marker = '*', s = 500)

# plt.title("2D Contour Plot of Trid Function")
plt.xlabel("x1",fontsize=18)
plt.ylabel("x2",fontsize=18)
plt.legend(fontsize = 14)
plt.tight_layout()