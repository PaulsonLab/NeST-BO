#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:56:37 2025

@author: tang.1856
"""


import torch
from LassoBench import LassoBench

class Leukemia():
    
    def __init__(self, negate=True):
        self.negate = negate
        # self.max_step = max_step
        # self.seed = seed
        self.fun = LassoBench.RealBenchmark(pick_data='leukemia')
    
    def __call__(self,x):
        x = x.to('cpu')
        y = torch.tensor([self.fun.evaluate(element) for element in x.numpy()])
        
        if self.negate:
            return -y
        else:
            return y
    
    
if __name__ == '__main__': 
    lb = -1
    ub = 1
    fun = Leukemia()
    x = torch.rand(2, 7129).to(torch.float64)
    y = fun(lb+(ub-lb)*x)