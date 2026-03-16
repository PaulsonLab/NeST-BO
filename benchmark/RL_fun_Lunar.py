#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 19:56:37 2025

@author: tang.1856
"""

import gymnasium as gym
import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Callable, Iterator, Union, Optional, List

class RL_fun():
    
    def __init__(self, negate=True, max_step = 300, mlp_shape=[10,2]):
        self.negate = negate
        self.max_step = max_step
        
    def policy(self, w, s):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])
    
        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]
    
        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]
    
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a
        
      
    
    def environment(self, param, seed=0):    
    
        env = gym.make("LunarLander-v3")    
        
        # options = {'goal_cell':np.array([5,2]), 'reset_cell':np.array([7,4])}
        observation, info = env.reset(seed=seed)
        # observation = self.manipulate_state(env.reset(seed=seed)[0])
       
        # initial_dist = np.linalg.norm(observation['achieved_goal']-observation['desired_goal'])
        Reward = 0
        for i in range(self.max_step):
           
            action = self.policy(param, observation)       
            observation, reward, terminated, truncated, info = env.step(action)
            # observation = self.manipulate_state(observation)
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
                Reward_total = -self.environment(element.numpy())
            else:
                Reward_total = self.environment(element.numpy())
            Reward_list.append(Reward_total) 
        
        return torch.tensor(Reward_list).to(torch.float32)
    
    
if __name__ == '__main__':    
    fun = RL_fun()
    x = torch.rand(2, 12)
    fun(x)