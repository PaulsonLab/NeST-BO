#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 13:32:57 2025

@author: tang.1856
"""

from optimization_loop_NeSTBO_sub import main
from botorch.test_functions.synthetic import Powell, Michalewicz, Griewank, Ackley
import torch
import math
import numpy as np
from dataclasses import dataclass
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64

def fun(x): # sphere function
    return torch.sum(x**2, dim=1) 

def embedding_matrix(input_dim: int, target_dim: int, seed) -> torch.Tensor:
    torch.manual_seed(seed)
    if (
        target_dim >= input_dim
    ):  # return identity matrix if target size greater than input size
        return torch.eye(input_dim, device=device, dtype=dtype)

    input_dims_perm = (
        torch.randperm(input_dim, device=device) + 1
    )  # add 1 to indices for padding column in matrix

    bins = torch.tensor_split(
        input_dims_perm, target_dim
    )  # split dims into almost equally-sized bins
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )  # zero pad bins, the index 0 will be cut off later

    mtrx = torch.zeros(
        (target_dim, input_dim + 1), dtype=dtype, device=device
    )  # add one extra column for padding
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )  # fill mask with random +/- 1 at indices

    return mtrx[:, 1:]  # cut off index zero as this corresponds to zero padding

@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = float("nan")  # Note: post-initialized
    target_dim: int = float("nan")  # Note: post-initialized
    n_splits: int = float("nan")  # Note: post-initialized
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = 1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        )
        self.d_init = max(self.d_init, target_dim_init)
        self.d_init = max(target_dim_init, self.d_init)
        self.target_dim = self.d_init
        self.n_splits = n_splits   
    
dim = 20
target_dim_init = 4
Ninit = 10
lb = -dim**2
ub = dim**2
delta = 0.5
Budget = 100
seed = 1

state = BaxusState(dim=dim, eval_budget=Budget)
target_dim = state.d_init

M = int(target_dim)
S = embedding_matrix(input_dim=dim, target_dim=target_dim, seed = seed)

torch.manual_seed(seed)
params = -1+2*torch.rand(1, target_dim).to(torch.float64)

alg = main(fun, seed, target_dim, Ninit, lb, ub, params, Budget, delta, M, S, state)
X, Y, min_obj_list = alg.exec_alg()



 
    