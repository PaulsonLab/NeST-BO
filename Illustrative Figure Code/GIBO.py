#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""
import sys

import os
sys.path.append(os.path.abspath('/home/tang.1856/Hessian/src'))
from botorch.test_functions.synthetic import Ackley, Rosenbrock, StyblinskiTang, Griewank
import torch
from Acquisition_GIBO import GradientInformation, optimize_acqf_custom_bo
from botorch.models.transforms import Normalize, Standardize
# from src.cholesky import one_step_cholesky
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from model import DerivativeExactGPSEModel
import gpytorch
import botorch 

def fun(X):
    
    term1 = torch.sum((X - 1) ** 2, dim=1)  # Sum of squared terms
    term2 = torch.sum(X[:, 1:] * X[:, :-1], dim=1)  # Sum of product of consecutive terms
    
    return term1 - term2  # Final Trid function value

dim = 2
Ninit = 8
lb = -dim**2
ub = dim**2
delta = 0.2
step = 0.25
epsilon_diff_acq_value = 0.2
max_samples_per_iteration = 4
bo_iter = max_samples_per_iteration + 1
replicate =1

regret_y, grad_norm, cost_list = [[] for _ in range(replicate)], [[] for _ in range(replicate)], [[] for _ in range(replicate)]
for seed in range(replicate):
    # params = torch.tensor([[0.9,-0.9,0.9,-0.9,0.9,-0.9]]).to(torch.float64)
    torch.manual_seed(seed)
    # params = torch.rand(1, dim).to(torch.float64)
    params = torch.tensor([[0.25,0.25]]).to(torch.float64)
    
    train_X = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(torch.float64)
    train_X = torch.cat((params, train_X))
    
    
    train_Y = fun(lb+(ub-lb)*train_X)
    
    regret_y[seed].append(float(min(train_Y)))
    
    for bo in range(bo_iter):
        print('iter = ', bo)
        bounds = torch.tensor([[-delta], [delta]]) + params
        bounds[bounds<0] = 0
        bounds[bounds>1] = 1
        bounds = bounds.to("cpu")
        
        gp = DerivativeExactGPSEModel(dim, ard_num_dims=dim, N_max = 5*dim)
        gp.append_train_data(train_X, train_Y)
        
        gp.posterior(
            params
        )  # Call this to update prediction strategy of GPyTorch.
        
        acquisition_fcn = GradientInformation(gp)
        # acquisition_fcn = GradientInformation(gp, MC_samples)
        
        # train GP
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            gp.likelihood, gp
        )
        try:
            botorch.fit.fit_gpytorch_mll(mll)
        except:
            print('cant fit GP')
        
        gp.posterior(
            params
        )  # Call this to update prediction strategy of GPyTorch.
        
        
        acquisition_fcn.update_theta_i(params) 
        
        # inner loop for GIBO (enhance gradient information)
        acq_value_old = None
        
        for i in range(max_samples_per_iteration):
            
            new_x, acq_value = optimize_acqf_custom_bo(acquisition_fcn, bounds, q=1, num_restarts =5, raw_samples = 20)
            new_y = fun(lb + (ub - lb) *new_x)
            
            if new_y.min() < train_Y.min():
                ind_best = new_y.argmin()               
                print(
                    f"New best query: {new_y[ind_best].item():.3f}"
                )
                
            train_X = torch.cat((new_x, train_X))
            train_Y = torch.cat((new_y, train_Y))
               
            gp.append_train_data(new_x, new_y)
            gp.posterior(params)
            acquisition_fcn.update_K_xX_dx()
            
            regret_y[seed].append(float(min(train_Y)))
                           
            # if acq_value_old is not None:
            #     diff = acq_value - acq_value_old
            #     if diff < epsilon_diff_acq_value:
            #         print(f"Stop sampling after {i+1} samples, since gradient certainty is {diff}.")
            #         break
                
            acq_value_old = acq_value
        
      
       
        mean_J, variance_J = gp.posterior_derivative(params) # gradient predicted by GP
        mean_H = gp.posterior_hessian(params)
        d = -(torch.inverse(mean_H[0]) @ mean_J[0]).detach().unsqueeze(0)
        
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach()
        # params = params - step * (torch.nn.functional.normalize(mean_J)*lengthscale).detach()
        params = params + d
        
        params[params<0] = 0
        params[params>1] = 1
        
        Y_next = torch.cat([fun(lb + (ub - lb) *params)])         
        
        train_X = torch.cat((params,train_X))
        train_Y = torch.cat((Y_next,train_Y))   
        
        
        regret_y[seed].append(float(min(train_Y)))
       
        
        if len(regret_y[seed])>=bo_iter:
            break
torch.save(train_X, 'GIBO.pt')