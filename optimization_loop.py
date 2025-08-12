#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:59:01 2025

@author: tang.1856
"""
import sys
import os
sys.path.append(os.path.abspath('/home/tang.1856/NeST-BO/src'))
import torch
from Acquisition_NeSTBO import GradientInformation, optimize_acqf_custom_bo
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from model import DerivativeExactGPSEModel
import gpytorch
import botorch 
# device = torch.device("cuda")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class main():
    
    def __init__(self, fun, seed, dim, Ninit, lb, ub, params, bo_iter, delta,
                 max_samples_per_iteration, epsilon_diff_acq_value, step):
        self.fun = fun
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.params = params
        self.bo_iter = bo_iter
        self.delta = delta
        self.max_samples_per_iteration = max_samples_per_iteration
        self.epsilon_diff_acq_value = epsilon_diff_acq_value
        self.step = step
        self.seed = seed
        
        self.train_X = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(torch.float64).to(device)
        self.train_X = torch.cat((self.params, self.train_X))    
        self.train_Y = fun(lb+(ub-lb)*self.train_X).detach().to(torch.float64).to(device)
    
    
    def is_positive_semi_definite_eigen(self, A):
        """Check if a tensor A is positive semi-definite using eigenvalues."""
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix is not square!")
        
        eigvals = torch.linalg.eigvalsh(A)  # Compute only real eigenvalues (Hermitian matrix)
        return torch.all(eigvals >= 0)
        
    def move_Newton(self, gp, mean_H, mean_J, sigma = 0.1, s = 1.0,  beta = 0.8):
       
        # do the Armijo line search
        d = -(torch.inverse(mean_H[0]) @ mean_J[0]).detach().unsqueeze(0) # descent direction

        f_current = gp.posterior(self.params).mean
        f_future = gp.posterior(self.params + s*d).mean
        Armijo = -sigma*s* (d @ mean_J.T)
        while f_current - f_future < Armijo and s>0.05:
            s*=beta
            f_future = gp.posterior(self.params + s*d).mean
            Armijo = -sigma*s* (d @ mean_J.T)
        self.params = self.params + s*d
        
        
    def move_GD(self, gp, mean_J, sigma = 0.1, s = 1,  beta = 0.5):
        
        # do the Armijo line search
        d = -(torch.nn.functional.normalize(mean_J)).detach() # descent direction
        # d = -mean_J.detach()
        
        f_current = gp.posterior(self.params).mean
        f_future = gp.posterior(self.params + s*d).mean
        Armijo = -sigma*s* (d @ mean_J.T)
        while f_current - f_future < Armijo and s>0.05:
            s*=beta
            f_future = gp.posterior(self.params + s*d).mean
            Armijo = -sigma*s* (d @ mean_J.T)
        self.params = self.params + s*d    
        
    def exec_alg(self):
        regret_y = [float(min(self.train_Y))]
            
        for bo in range(self.bo_iter):
            
            print('Outer loop iter = ', bo)
            bounds = torch.tensor([[-self.delta], [self.delta]]).to(device) + self.params
            bounds[bounds<0] = 0
            bounds[bounds>1] = 1
            bounds = bounds.to("cpu")
            
            gp = DerivativeExactGPSEModel(self.dim, ard_num_dims=self.dim, N_max=100*self.dim)
            gp = gp.to(device)
            gp.append_train_data(self.train_X, self.train_Y)
            
            gp.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.
            
            
            # train GP
            mll = ExactMarginalLogLikelihood(
                gp.likelihood, gp
            )
            try:
                botorch.fit.fit_gpytorch_mll(mll)
            except:
                print('cant fit GP')
            print('GP fitted lscale=',gp.covar_module.base_kernel.lengthscale[0])
            
            gp.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.
            
            acquisition_fcn = GradientInformation(gp)
            acquisition_fcn.update_theta_i(self.params) 
            
            # inner loop for GIBO (enhance gradient information)
            acq_value_old = None
            
            for i in range(self.max_samples_per_iteration):
                
                new_x, acq_value = optimize_acqf_custom_bo(acquisition_fcn, bounds, q = 1, num_restarts = 10, raw_samples = 200)
                new_y = self.fun(self.lb + (self.ub - self.lb) *new_x).detach().to(torch.float64).to(device)
                    
                self.train_X = torch.cat((new_x, self.train_X))
                self.train_Y = torch.cat((new_y, self.train_Y))
                regret_y.append(float(min(self.train_Y)))
                if len(regret_y)>=self.bo_iter:
                    break
                
                gp.append_train_data(new_x, new_y)
                gp.posterior(self.params)
                acquisition_fcn.update_K_xX_dx()
                
                if acq_value_old is not None:
                    diff = acq_value - acq_value_old
                    
                    if diff < self.epsilon_diff_acq_value:
                        print(f"Stop sampling after {i+1} samples, since gradient certainty is {diff}.")
                        break
                    
                acq_value_old = acq_value
                
            
            # train GP
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                gp.likelihood, gp
            )
            try:
                botorch.fit.fit_gpytorch_mll(mll)
            except:
                print('cant fit GP')
            
            gp.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.
            
            mean_J, variance_J = gp.posterior_derivative(self.params) # gradient predicted by GP
            mean_H = gp.posterior_hessian(self.params)
           
            if self.is_positive_semi_definite_eigen(mean_H[0]):
                print('PSD!!')
                self.move_Newton(gp, mean_H, mean_J)
            else:    
                self.move_GD(gp, mean_J)
            
    
            self.params[self.params<0] = 0
            self.params[self.params>1] = 1
            
            
            Y_next = torch.cat([self.fun(self.lb + (self.ub - self.lb) *self.params)]).detach().to(torch.float64)   
            
            self.train_X = torch.cat((self.params, self.train_X))
            self.train_Y = torch.cat((Y_next, self.train_Y))  
                 
            regret_y.append(float(min(self.train_Y)))
                   
            print('regret=', regret_y[-1])
           
            if len(regret_y)>=self.bo_iter:
                break
            
        return self.train_X, self.train_Y, regret_y