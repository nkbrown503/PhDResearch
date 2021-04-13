# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""


''' Nathan Brown 
Policy Gradient Training of Topology Optimization through Reinforcement learning'''

import gym
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import deque
import time
import random
from gym import Env
from gym.spaces import Discrete, Box
from random import choice
import math
import FEA_SOLVER_V
#Define The Size and Scope of Your Training
Lx=1
Ly=1
Elements_X=6
Elements_Y=6
ElementSize=Elements_X*Elements_Y
Vol_fraction=16/36
Remove_num=ElementSize-(ElementSize*Vol_fraction)

#---Loading State Table----------------
LeftBC=list(range(0,Elements_X*(Elements_Y-1)*2,Elements_X*(Elements_Y-1)))
LeftBC.append(Elements_X-1) #For Bottom Right
class TopOpt_6by6_V(Env):
    def __init__(self):
        #Actons we can take... remove any of the blocks
        self.action_space=Discrete(ElementSize)
        #Current Material Matrix
        Low=np.array([0]*ElementSize)
        High=np.array([1]*ElementSize)
        self.observation_space=Box(Low,High,dtype=np.float32)
        
        # Set Starting Material Matrix
        VoidCheck=np.ones((1,ElementSize))
        VoidCheck=list(VoidCheck[0])
        self.state = VoidCheck
        # Set Maximum Number of Blocks to Remove
        self.Remove_Tot=Remove_num
        
    def step(self,action):
        #Apply Action
        '''Remove any blocks, but if it removes a boundary condition
        or a block that leads to a non-singular body, then the agent 
        should be penalized'''
        
        rs_place=self.state[action]
        self.state[action]=0
        ElementMat=np.reshape(self.state,(Elements_X,Elements_Y))
        SingleCheck=FEA_SOLVER_V.isolate_largest_group_original(ElementMat)
        It=list(self.state).count(0)
        if rs_place==1 and action not in LeftBC and SingleCheck[1]==True:
            if It>=self.Remove_Tot:
                done=True 
            else:
                done=False
                
            reward=(FEA_SOLVER_V.FEASolve(list(self.state),Lx,Ly,Elements_X,Elements_Y)[1])*(-1*10**-3)
            reward=(1+reward[0][0])*It
           
    
        else: 
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-100)"""
            reward=-1
            done=True
            
        #PLaceholder for Info
        info={}
            
        return self.state, reward, done, info
    
    def render(self, mode='human'):
        ElementMat=np.reshape(self.state,(Elements_X,Elements_Y))
        print(ElementMat)
        print('')
        
    def reset(self):
        VoidCheck=np.ones((1,ElementSize))
        VoidCheck=list(VoidCheck[0])
        self.state = VoidCheck
        self.state=np.array(self.state)
        return self.state
      

env = TopOpt_6by6_V()
env.Test=False 
env.Loc_List=[]
env.Dir_List=[]
env.Rem_List=[]
# An episode a full game
train_episodes = 300

def create_actor(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])

    return model

def create_critic(state_shape, output_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def main():
    actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "training_critic/critic_cp.ckpt"

    actor = create_actor(env.observation_space.shape, env.action_space.n)
    critic = create_critic(env.observation_space.shape, 1)
    if os.path.exists('training_actor'):
        actor.load_weights(actor_checkpoint_path)

        critic.load_weights(critic_checkpoint_path)

    # X = states, y = actions
    X = []
    y = []

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:

            # model dims are (batch, env.observation_space.n)
            observation_reshaped = observation.reshape([1, observation.shape[0]])
            action_probs = actor.predict(observation_reshaped).flatten()
            # Note we're sampling from the prob distribution instead of using argmax
            action = np.random.choice(env.action_space.n, 1, p=action_probs)[0]
            encoded_action = one_hot_encode_action(action, env.action_space.n)

            next_observation, reward, done, info = env.step(action)
            next_observation_reshaped = next_observation.reshape([1, next_observation.shape[0]])

            value_curr = np.asscalar(np.array(critic.predict(observation_reshaped)))
            value_next = np.asscalar(np.array(critic.predict(next_observation_reshaped)))

            # Fit on the current observation
            discount_factor = .7
            TD_target = reward + (1 - done) * discount_factor * value_next
            advantage = critic_target = TD_target - value_curr
            print(np.around(action_probs, 2),next_observation, np.around(value_next - value_curr, 3), 'Advantage:', np.around(advantage, 2))
            advantage_reshaped = np.vstack([advantage])
            TD_target = np.vstack([TD_target])
            critic.train_on_batch(observation_reshaped, TD_target)
            #critic.fit(observation_reshaped, TD_target, verbose=0)

            gradient = encoded_action - action_probs
            gradient_with_advantage = .0001 * gradient * advantage_reshaped + action_probs
            actor.train_on_batch(observation_reshaped, gradient_with_advantage)
            #actor.fit(observation_reshaped, gradient_with_advantage, verbose=0)
            observation = next_observation
            total_training_rewards += reward

            if done:
                print('Ep: {} Ep. Reward: {} Ep. Steps = {} Final Reward: {} '.format(episode,round(total_training_rewards,3), list(env.state).count(0), round(reward,3)))
                total_training_rewards += 1

                actor.save_weights(actor_checkpoint_path)
                critic.save_weights(critic_checkpoint_path)

    

if __name__ == '__main__':
    main()
    
LL_plot=[]
DL_plot=[]
RL_plot=[]
LL_Label=[]
DL_Label=['Compression','Tension']
RL_Label=[]

for j in range(min(env.Loc_List),max(env.Loc_List)):
    LL_plot.append(env.Loc_List.count(j))
    
DL_plot.append(env.Dir_List.count(-1))
DL_plot.append(env.Dir_List.count(1))
    
for j in range(min(env.Rem_List),max(env.Rem_List)):
    RL_plot.append(env.Rem_List.count(j))
    
for j in range(min(env.Loc_List),max(env.Loc_List)):
    LL_Label.append(str(j))
    
for j in range(min(env.Rem_List),max(env.Rem_List)):
    RL_Label.append(str(j))
           
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(LL_Label,LL_plot)
plt.xlabel('Loaded DOF')
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(DL_Label,DL_plot)
plt.show()

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(RL_Label,RL_plot)
plt.xlabel('Remaining Blocks')
plt.show()
