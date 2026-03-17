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
    
    def __init__(self, negate=True, max_step = 1000, seed = 1):
        self.negate = negate
        self.max_step = max_step
        self.seed = seed

    def policy(self, param, state):
       
        param_matrix = param.reshape(8, 111)
        p = param_matrix @ state
      
        p[p>1] = 1
        p[p<-1] = -1

        return p.numpy()
    
    def environment(self, param):    
    
        env = gym.make("Ant-v4", exclude_current_positions_from_observation=True, use_contact_forces=True)    
        
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
        x = x.to('cpu')
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
    x = -1+2*torch.rand(1, 888)
    # x = 0.5*torch.ones(102).unsqueeze(0)
    reward = fun(x)