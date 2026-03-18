#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:59:01 2025

@author: tang.1856
"""
# import sys
# import os
import logging
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
# sys.path.append(os.path.abspath('/fs/ess/PAS2983/jontwt/NeST-BO/src'))
import torch
from src.Acquisition_NeSTBO import NewtonInformation, optimize_acqf_custom_bo
from gpytorch.mlls import ExactMarginalLogLikelihood
from model import DerivativeExactGPSEModel
import gpytorch
import botorch 
import hydra
import tqdm as tqdm
from gpytorch.priors.torch_priors import GammaPrior
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

class main():
    
    def __init__(self, config: DictConfig) -> None:
        # Resolve configuration
        OmegaConf.resolve(config)
        logging.info("\n" + OmegaConf.to_yaml(config))
        
        self.seed = config.seed
        self.device = config.device
        self.T = config.benchmark.n_tot
        self.delta = config.benchmark.delta
        self.M = config.benchmark.M
        # self.N_max = config.benchmark.N_max
        self.fun = hydra.utils.instantiate(config.benchmark.fn)
        try:
            self.fun = self.fun.to(dtype).to(self.device)
        except:
            None
        self.dim = config.benchmark.dim
        self.lb = torch.tensor(config.benchmark.lb).to(dtype).to(self.device)
        self.ub = torch.tensor(config.benchmark.ub).to(dtype).to(self.device)
        self.N_init = config.benchmark.N_init
        if config.benchmark.params.random:
            torch.manual_seed(self.seed)
            self.params = torch.rand(self.dim, dtype=dtype, device=self.device).unsqueeze(0)
        elif config.benchmark.params.center:
            self.params = torch.tensor([0.5]*self.dim).unsqueeze(0).to(dtype).to(self.device)
        else:
            self.params = torch.tensor([config.benchmark.params.init], dtype=dtype, device=self.device)
        self.train_X = torch.quasirandom.SobolEngine(dimension=self.dim,  scramble=True, seed=self.seed).draw(self.N_init).to(dtype).to(self.device)
        self.train_X = torch.cat((self.params, self.train_X))    
        self.train_Y = self.fun(self.lb+(self.ub-self.lb)*self.train_X).detach().to(dtype).to(self.device)
    
    def is_positive_semi_definite_eigen(self, A):
        """Check if a tensor A is positive semi-definite using eigenvalues."""
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix is not square!")
        
        eigvals = torch.linalg.eigvalsh(A)  # Compute only real eigenvalues (Hermitian matrix)
        return torch.all(eigvals >= 0)
        
    def move_Newton(self, gp, mean_H, mean_J, sigma = 0.1, s = 1.0,  beta = 0.8):
       
        # do the Armijo line search
        d = -(torch.inverse(mean_H[0]) @ mean_J[0]).detach().unsqueeze(0) # Newton-step

        f_current = gp.posterior(self.params).mean
        f_future = gp.posterior(self.params + s*d).mean
        Armijo = -sigma*s* (d @ mean_J.T)
        while f_current - f_future < Armijo and s > 0.05:
            s*=beta
            f_future = gp.posterior(self.params + s*d).mean
            Armijo = -sigma*s* (d @ mean_J.T)
            
        self.params = self.params + s*d
           
    def move_GD(self, gp, mean_J, sigma = 0.1, s = 0.5,  beta = 0.5):
        
        # do the Armijo line search
        d = -(torch.nn.functional.normalize(mean_J)).detach() # descent direction
        
        f_current = gp.posterior(self.params).mean
        f_future = gp.posterior(self.params + s*d).mean
        Armijo = -sigma*s* (d @ mean_J.T)
        while f_current - f_future < Armijo and s > 0.05:
            s*=beta
            f_future = gp.posterior(self.params + s*d).mean
            Armijo = -sigma*s* (d @ mean_J.T)
            
        self.params = self.params + s*d    
        
    def exec_alg(self):
        
        lengthscale_constraint=gpytorch.constraints.Interval(0.005, 10)
        outputscale_constraint=None
        
        regret_y = [float(min(self.train_Y))]
        
        y_min: float = self.train_Y.min().item()

        # Construct iterator for BO loop
        tqdm_log_list = ["y_min"]
        pbar = tqdm.tqdm(
            initial=1,
            total=self.T,
            desc="# function evaluation",
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            disable=(not len(tqdm_log_list)),
        )
            
        for bo in range(self.T):
            
            # print('Outer loop iter:', bo)
            bounds = torch.tensor([[-self.delta], [self.delta]]).to(self.device) + self.params
            bounds[bounds<0] = 0
            bounds[bounds>1] = 1
            bounds = bounds.to(self.device)
            
            gp = DerivativeExactGPSEModel(self.dim, ard_num_dims=self.dim, lengthscale_constraint=lengthscale_constraint, outputscale_constraint=outputscale_constraint)
            gp = gp.to(self.device)
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
                
            # print('GP fitted lscale=',gp.covar_module.base_kernel.lengthscale[0])
            # print('GP fitted output scale=',gp.covar_module.outputscale)
            
            gp.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.
            
            acquisition_fcn = NewtonInformation(gp)
            acquisition_fcn.update_theta_i(self.params) 
            
            # inner loop for NeST-BO 
            
            for i in range(self.M):
                # print('Inner loop iter:', i)
                new_x, acq_value = optimize_acqf_custom_bo(acquisition_fcn, bounds, q = 1, num_restarts = 5, raw_samples = 20)
                new_y = self.fun(self.lb + (self.ub - self.lb) *new_x).detach().to(dtype).to(self.device)
                    
                self.train_X = torch.cat((new_x, self.train_X))
                self.train_Y = torch.cat((new_y, self.train_Y))
                regret_y.append(float(min(self.train_Y)))
                pbar.update(1)
                gp.append_train_data(new_x, new_y)
                gp.posterior(self.params)
                acquisition_fcn.update_K_xX_dx()
                
                if len(regret_y)>=self.T:
                    break
            
            if len(regret_y)>=self.T:
                break
            
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
            
            # IsPD or not
            if self.is_positive_semi_definite_eigen(mean_H[0]):
                # print('PSD!')
                self.move_Newton(gp, mean_H, mean_J)
            else:    
                self.move_GD(gp, mean_J)
            
            self.params[self.params<0] = 0
            self.params[self.params>1] = 1
            
            
            Y_next = torch.cat([self.fun(self.lb + (self.ub - self.lb) *self.params)]).detach().to(dtype).to(self.device)  
            
            self.train_X = torch.cat((self.params, self.train_X))
            self.train_Y = torch.cat((Y_next, self.train_Y))  
                 
            regret_y.append(float(min(self.train_Y)))
                   
            # print('min obj value =', regret_y[-1])
                   
            with torch.no_grad():
                # See if we have a new best observation
                y_curr = self.train_Y.min().item()
                
                if y_curr < y_min:
                    y_min = y_curr

                # Update progress bar
                iter_stats = OrderedDict(
                    y_min=y_min,
                    y_curr=y_curr,
                )
                pbar.set_postfix(**{stat: iter_stats.get(stat, None) for stat in tqdm_log_list})

            pbar.update(1)
            
            if len(regret_y)>=self.T:
                break

        pbar.close()

        logging.info(f"min obj value: {y_min}")
            
        return self.train_X, self.train_Y, regret_y

# def __init__(self, fun, seed, dim, Ninit, lb, ub, params, T, delta, M):
# self.fun = fun
# self.dim = dim
# self.lb = lb
# self.ub = ub
# self.params = params.to(device)
# self.T = T
# self.delta = delta
# self.M = M
# self.seed = seed
# self.train_X = torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(torch.float64).to(device)
# self.train_X = torch.cat((self.params, self.train_X))    
# self.train_Y = fun(lb+(ub-lb)*self.train_X).detach().to(torch.float64).to(device)
