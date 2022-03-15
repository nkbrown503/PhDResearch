 # -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:49:00 2022

@author: nbrow
"""
from agent import PPOAgent
import gym
import numpy as np

import os
from UC_Env import UC_Env
#tf.set_random_seed(0)
if __name__ == "__main__":
    # newest gym fixed bugs in 'BipedalWalker-v2' and now it's called 'BipedalWalker-v3'
    env = UC_Env()
    agent = PPOAgent(env)
    agent.run_batch() # train as PPO
    #agent.run_multiprocesses(num_worker = 16)  # train PPO multiprocessed (fastest)
    #agent.test()