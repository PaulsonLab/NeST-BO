#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 19:59:01 2025

@author: tang.1856
"""
import sys
import os
sys.path.append(os.path.abspath('/fs/ess/PAS2983/jontwt/NeST-BO/src'))
import torch
from Acquisition_NeSTBO import NewtonInformation, optimize_acqf_custom_bo
from gpytorch.mlls import ExactMarginalLogLikelihood
from model import DerivativeExactGPSEModel
import gpytorch
import botorch 
import math

dtype = torch.float64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def update_state(state, Y_next):
    if -max(Y_next) > -state.best_value:
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    state.best_value = -max(-state.best_value, -max(Y_next).item())

    return state

def increase_embedding_and_observations(
    S: torch.Tensor, X: torch.Tensor, n_new_bins: int
) -> torch.Tensor:
    assert X.size(1) == S.size(0), "Observations don't lie in row space of S"

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].reshape(-1)

        if len(idxs_non_zero) <= 1:
            continue

        non_zero_elements = row[idxs_non_zero].reshape(-1)

        n_row_bins = min(
            n_new_bins, len(idxs_non_zero)
        )  # number of new bins is always less or equal than the contributing input dims in the row minus one

        new_bins = torch.tensor_split(idxs_non_zero, n_row_bins)[
            1:
        ]  # the dims in the first bin won't be moved
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        new_bins_padded = torch.nn.utils.rnn.pad_sequence(
            new_bins, batch_first=True
        )  # pad the tuples of bins with zeros to apply _scatter
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), device=device, dtype=dtype
        )  # submatrix to stack on S_update

        S_stack = S_stack.scatter_(
            1, new_bins_padded + 1, els_to_move_padded
        )  # fill with old values (add 1 to indices for padding column)

        S_update[
            row_idx, torch.hstack(new_bins)
        ] = 0  # set values that were move to zero in current row

        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)))
        )  # repeat observations for row at the end of X (column-wise)
        S_update = torch.vstack(
            (S_update, S_stack[:, 1:])
        )  # stack onto S_update except for padding column

    return S_update, X_update

class main():
    
    def __init__(self, fun, seed, dim, Ninit, lb, ub, params, bo_iter, delta, M, S, state):
        self.fun = fun
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.params = params.to(device)
        self.bo_iter = bo_iter
        self.delta = delta
        self.M = M
        self.seed = seed
        self.S = S
        self.state = state
        self.train_X = -1+2*torch.quasirandom.SobolEngine(dimension=dim,  scramble=True, seed=seed).draw(Ninit).to(torch.float64).to(device)
        self.train_X = torch.cat((self.params, self.train_X))       
        train_X_inverse = (self.train_X @ self.S + 1)/2
        self.train_Y = fun(lb+(ub-lb)*(train_X_inverse)).detach().to(torch.float64).to(device)
       
    
    def is_positive_semi_definite_eigen(self, A):
        """Check if a tensor A is positive semi-definite using eigenvalues."""
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix is not square!")
        
        eigvals = torch.linalg.eigvalsh(A)  # Compute only real eigenvalues (Hermitian matrix)
        return torch.all(eigvals >= 0)
    
        
    def move_Newton(self, gp, mean_H, mean_J, sigma = 0.1, s = 1.0,  beta = 0.5):
     
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
        
        
    def move_GD(self, gp, mean_J, sigma = 0.1, s = 0.5,  beta = 0.5):
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach()
        # do the Armijo line search
        d = -(torch.nn.functional.normalize(mean_J)*lengthscale).detach()
        
        f_current = gp.posterior(self.params).mean
        f_future = gp.posterior(self.params + s*d).mean
        Armijo = -sigma*s* (d @ mean_J.T)
        while f_current - f_future < Armijo and s>0.05:
            s*=beta
            f_future = gp.posterior(self.params + s*d).mean
            Armijo = -sigma*s* (d @ mean_J.T)
        self.params = self.params + s*d   
        
        
    def exec_alg(self):
        regret_y = []
       
        regret_y.append(float(min(self.train_Y)))
        
        for bo in range(self.bo_iter):
           
                
            print('Outler loop iter:', bo)
            bounds = torch.tensor([[-self.delta], [self.delta]]).to(device) + self.params
            bounds[bounds<-1] = -1
            bounds[bounds>1] = 1
            bounds = bounds.to(device)
            
            gp = DerivativeExactGPSEModel(self.dim, ard_num_dims=self.dim)
            gp = gp.to(device)
            gp.append_train_data(self.train_X, self.train_Y)
            
            gp.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.
            

            # train GP
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                gp.likelihood, gp
            )
            try:
                botorch.fit.fit_gpytorch_mll(mll)
            except:
                print('cant fit GP')
                
            print('lscale=',gp.covar_module.base_kernel.lengthscale)
            gp.posterior(
                self.params
            )  # Call this to update prediction strategy of GPyTorch.
            
            acquisition_fcn = NewtonInformation(gp)
            acquisition_fcn.update_theta_i(self.params) 
            
            # inner loop for NeST-BO-sub         
            for i in range(self.M):
                print('Inner loop iter:', i)
                new_x, acq_value = optimize_acqf_custom_bo(acquisition_fcn, bounds, q=1, num_restarts = 5, raw_samples = 20)
                new_x_inverse = (new_x @ self.S + 1)/2
                new_y = self.fun(self.lb + (self.ub - self.lb) *new_x_inverse).detach().to(torch.float64).to(device)
                
                self.train_X = torch.cat((new_x, self.train_X))
                self.train_Y = torch.cat((new_y, self.train_Y))
                regret_y.append(float(min(self.train_Y)))
                if len(regret_y)>=self.bo_iter:
                    break
                
                gp.append_train_data(new_x, new_y)
                gp.posterior(self.params)
                acquisition_fcn.update_K_xX_dx()
                
            
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
                self.move_Newton(gp, mean_H, mean_J)
            else:    
                self.move_GD(gp, mean_J)
            
            self.params[self.params<-1] = -1
            self.params[self.params>1] = 1
            
            params_inverse = (self.params @ self.S + 1)/2
            Y_next = torch.cat([self.fun(self.lb + (self.ub - self.lb) *params_inverse)]).detach().to(torch.float64)   
             
            self.train_X = torch.cat((self.params, self.train_X))
            self.train_Y = torch.cat((Y_next, self.train_Y))  
            self.state = update_state(state=self.state, Y_next=min(self.train_Y[0:i+2]).unsqueeze(0))
            
            regret_y.append(float(min(self.train_Y)))
          
            print('min obj value =', regret_y[-1])
           
            if len(regret_y)>=self.bo_iter:
                break
            
            if self.state.failure_counter == 10:
                self.state.restart_triggered = False
                print("increasing target space")
                self.S, self.train_X = increase_embedding_and_observations(
                    self.S, self.train_X, self.state.new_bins_on_split
                )
                self.params = self.train_X[0].unsqueeze(0)
                print(f"new dimensionality: {len(self.S)}")
                self.state.target_dim = len(self.S)
                self.dim = len(self.S)
                self.max_samples_per_iteration = self.dim
                self.state.failure_counter = 0
                self.state.success_counter = 0
            
            
        return self.train_X, self.train_Y, regret_y