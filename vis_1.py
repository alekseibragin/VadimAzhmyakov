# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:46:25 2021

@author: Vadim
"""

import gym
env = gym.make('CarRacing-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())