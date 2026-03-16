#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:56:37 2025

@author: tang.1856
"""

import gymnasium as gym
import torch
import numpy as np

class RL_fun():
    
    def __init__(self, negate=True, max_step = 1000, seed = 0):
        self.negate = negate
        self.max_step = max_step
        self.seed = seed

    def policy(self, param, state):
        
        # policy for swimmer
        p1 = param[0]*state[0] + param[1]*state[1] + param[2]*state[2] + param[3]*state[3] + param[4]*state[4] + param[5]*state[5] + param[6]*state[6] + param[7]*state[7] 
        p2 = param[8]*state[0] + param[9]*state[1] + param[10]*state[2] + param[11]*state[3] + param[12]*state[4] + param[13]*state[5] + param[14]*state[6] + param[15]*state[7] 
        # p1 = -1+2*float(torch.sigmoid(p1))
        # p2 = -1+2*float(torch.sigmoid(p2))
        # if p1>1.0:
        #     p1 = 1.0
        # elif p1<-1.0:
        #     p1 = -1.0
        
        # if p2>1.0:
        #     p2 = 1.0
        # elif p2<-1.0:
        #     p2 = -1.0
        return np.array([p1,p2])
    
    def environment(self, param):    
    
        env = gym.make("Swimmer-v5")    
        
        # options = {'goal_cell':np.array([5,2]), 'reset_cell':np.array([7,4])}
        observation, info = env.reset(seed=self.seed)
       
        # initial_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
        Reward = 0
        for i in range(self.max_step):
           
            action = self.policy(param, observation)       
            observation, reward, terminated, truncated, info = env.step(action)
            Reward+=reward
            if terminated or truncated:
              # observation['achieved_goal'] = observation['desired_goal']
              break      
        env.close()
        return Reward

    def __call__(self,x):
        
        Reward_list = []
        for element in x:
            if self.negate:                       
                Reward_total = -self.environment(element)
            else:
                Reward_total = self.environment(element)
            Reward_list.append(Reward_total) 
        
        return torch.tensor(Reward_list).to(torch.float32)
    
    
if __name__ == '__main__':    
    fun = RL_fun()
    x = torch.rand(2, 16)
    fun(x)