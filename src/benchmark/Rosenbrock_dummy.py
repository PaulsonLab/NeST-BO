#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:56:37 2025

@author: tang.1856
"""
import torch
from botorch.test_functions.synthetic import Rosenbrock

class Rosenbrock_Dummy():
    
    def __init__(self, dim_true, negate=False):
        self.dim_true = dim_true
        self.f = Rosenbrock(dim=dim_true, negate=negate)

    def __call__(self,x):
        self.f.to(x.device).to(x.dtype)
        return self.f(x[..., :self.dim_true])
    
    
if __name__ == '__main__':    
    fun =Rosenbrock_Dummy(dim_true = 3)
    x = torch.rand(2, 16)
    fun(x)