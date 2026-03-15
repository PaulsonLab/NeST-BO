#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:27:25 2025

@author: tang.1856
"""

from typing import Tuple
import torch
import botorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradientInformation(botorch.acquisition.AnalyticAcquisitionFunction):
    """Acquisition function to sample points for gradient information.

    Attributes:
        model: Gaussian process model that supplies the Jacobian (e.g. DerivativeExactGPSEModel).
    """

    def __init__(self, model):
        """Inits acquisition function with model."""
        super().__init__(model)
        # pre-calculate the trace of the fourth-order derivative matrix
        lscale = self.model.covar_module.base_kernel.lengthscale.detach().to(torch.float64).flatten()
        sigma_f = self.model.covar_module.outputscale.detach()
        self.trace_fourth_derivative = []
        for i in range(self.model.D):
            for j in range(self.model.D):
                if i==j:
                    self.trace_fourth_derivative.append(sigma_f*3/(lscale[i]**2*lscale[j]**2))
                else:
                    self.trace_fourth_derivative.append(sigma_f/(lscale[i]**2*lscale[j]**2))
        self.trace_fourth_derivative = torch.tensor(self.trace_fourth_derivative,device=device)

    def update_theta_i(self, theta_i: torch.Tensor):
        """Updates the current parameters.

        This leads to an update of K_xX_dx.

        Args:
            theta_i: New parameters.
        """
        if not torch.is_tensor(theta_i):
            theta_i = torch.tensor(theta_i)
        self.theta_i = theta_i
        self.update_K_xX_dx()

    def update_K_xX_dx(self):
        """When new x is given update K_xX_dx."""
        # Pre-compute large part of K_xX_dx.
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, self.model.D).to(torch.float64)
        # x = self.theta_i.view(-1, self.model.D)
        self.K_xX_dx_part = self._get_KxX_dx(x, X)

    def _get_KxX_dx(self, x, X) -> torch.Tensor:
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.

        Args:
            x: (n x D) Test points.

        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X = X.to(torch.float64)
        x = x.to(torch.float64)
        N = X.shape[0]
        n = x.shape[0]
        K_xX = self.model.covar_module(x, X).evaluate()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        
        return (
            -torch.eye(self.model.D, device=X.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.model.D) - X.view(1, N, self.model.D))
                * K_xX.view(n, N, 1)
            ).transpose(1, 2)
        )
    
    def _get_KxX_dxdx(self, x, X):
        
        X = X.to(torch.float64)
        x = x.to(torch.float64)
        N = X.shape[0]
        n = x.shape[0]
        K_xX = self.model.covar_module(x, X).evaluate()
        dk_dx = self._get_KxX_dx(x, X)
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach().to(torch.float64)
        L = torch.eye(self.model.D, device=lengthscale.device) / lengthscale ** 2
        x1_x2 = (x.view(n, 1, self.model.D) - X.view(1, N, self.model.D)).transpose(1, 2)
                 
        k_expanded = K_xX.unsqueeze(1).unsqueeze(2)
        first_term = -k_expanded * L.unsqueeze(0).unsqueeze(-1)
        
        A = (-L @ x1_x2).unsqueeze(3)
        B = dk_dx.transpose(1,2).unsqueeze(1)
        second_term = (A*B).transpose(-1,-2)
     
        return first_term+second_term
    
    @botorch.utils.transforms.t_batch_mode_transform(expected_q=1)
    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function on the candidate set thetas.

        Args:
            thetas: A (b) x D-dim Tensor of (b) batches with a d-dim theta points each.

        Returns:
            A (b)-dim Tensor of acquisition function values at the given theta points.
        """
        sigma_n = self.model.likelihood.noise_covar.noise
        D = self.model.D
        X = self.model.train_inputs[0]
        x = self.theta_i.view(-1, D)
        
        n = x.shape[0]
        

        variances = []
        # torch.set_float32_matmul_precision("medium")
        for theta in thetas.to(device):
            
            theta = theta.view(-1, D)
            
            X_look_head = torch.cat((X, theta))
            N = X_look_head.shape[0]
            K_xX_dx = self._get_KxX_dx(x, X_look_head)
            K_xX_dxdx = self._get_KxX_dxdx(x, X_look_head)
            K_XX_inv = torch.inverse((self.model.covar_module(X_look_head, X_look_head).evaluate()) + sigma_n * torch.eye(X_look_head.shape[0], device=device))
           
        
            # variance_d = self.model._get_Kxx_dx2()-K_xX_dx @ K_XX_inv.to(torch.float64) @ K_xX_dx.transpose(1, 2) # look ahead gradient variance
            variance_d = -K_xX_dx @ K_XX_inv.to(torch.float64) @ K_xX_dx.transpose(1, 2)
            # variance_H = - ((K_xX_dxdx @ K_XX_inv).view(n,D*D, N) @ K_xX_dxdx.view(n,D*D, N).transpose(1,2))
            # row_idx = torch.arange(D).repeat(D)
            # col_idx = torch.arange(D).repeat_interleave(D)
            # trace_H = variance_H[0, row_idx * D + row_idx, col_idx * D + col_idx]
            
            # variance_H = torch.zeros(D,D,D,D)
            # for i in range(D):
            #     for j in range(D):
            #         for k in range(N):
            #             variance_H[i,j] += (K_xX_dxdx @ K_XX_inv).squeeze(0)[:,:,k][i,j]*K_xX_dxdx.squeeze(0).transpose(0,2)[k] 
                    
            A = torch.matmul(K_xX_dxdx, K_XX_inv).squeeze(0)   # (D, D, N)
            B = K_xX_dxdx.squeeze(0).permute(2, 0, 1)          # (N, D, D)
            variance_H = torch.tensordot(A, B, dims=([2], [0]))  # (D, D, D, D)
            idx_i = torch.arange(D)
            idx_j = torch.arange(D)
            trace_H = -variance_H[idx_i[:, None], idx_j[None, :], idx_i[:, None], idx_j[None, :]].flatten()
        
            
            
            #trace_H = self.trace_fourth_derivative + variance_H[0, row_idx * D + row_idx, col_idx * D + col_idx]
            variances.append(torch.trace(variance_d.view(D, D)).view(1) + torch.sum(trace_H).view(1))

        return -torch.cat(variances, dim=0)
    
    
def optimize_acqf_custom_bo(
    acq_func: botorch.acquisition.AcquisitionFunction,
    bounds: torch.Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Function to optimize the GradientInformation acquisition function for custom Bayesian optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A 2 x D tensor of lower and upper bounds for each column of X.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition function optimization.
        raw_samples: The number of samples for initialization.

    Returns:
        A two-element tuple containing:
            - a q x D-dim tensor of generated candidates.
            - a tensor of associated acquisition values.
    """
    # torch.manual_seed(1)
    
    candidates, acq_value = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,  # Analytic acquisition function.
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # Used for initialization heuristic.
        options={"nonnegative": True, "batch_limit": 5},
        return_best_only=True,
        sequential=False,
        # timeout_sec = 1
    )
    # Observe new values.
    new_x = candidates.detach().to(device)
    return new_x, acq_value