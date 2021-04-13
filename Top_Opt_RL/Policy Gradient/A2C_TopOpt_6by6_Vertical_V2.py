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
import tensorflow_probability as tfp
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow import keras
import pandas as pd
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
tic=time.perf_counter()

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
            if reward<0:
                reward=0
    
        else:
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-100)"""
            reward=-1
            done=True
            
        #PLaceholder for Info
        info={}
            
        return np.array(self.state), reward, done, info
    
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
      
class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
            name='actor_critic_6b6_V_R2_V2', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi
    
class Agent:
    def __init__(self, alpha=0.001, gamma=0.99, n_actions=2):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()
        log_prob = action_probabilities.log_prob(action)
        self.action = action

        return action.numpy()[0]

    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
        
    def learn(self, state,action, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32) # not fed to NN
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(self.action)

            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss = -log_prob*delta
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))
        
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episodes')
    plt.ylabel(' Average Reward')
    plt.savefig(figure_file)
        
#env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0')
agent = Agent(alpha=1e-4, n_actions=env.action_space.n)
n_games = 300_000
# uncomment this line and do a mkdir tmp && mkdir video if you want to
# record video of the agent playing the game.
#env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
filename = 'TopOpt_6by6_V2_R2_reward.png'

figure_file = 'plots/' + filename

best_score = env.reward_range[0]
score_history = []
step_history=[]
per_history=[]
succ_history=[]

load_checkpoint = False 
Retrain=True
if load_checkpoint:
    agent.load_models()
if Retrain is False:
    TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Time'])
if Retrain is True:
    agent.load_models()
    TrialData=pd.read_pickle('TopOpt_6b6_V_V2_R2_TrialData.pkl')

for i in range(n_games):
    observation = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        score += reward
        if not load_checkpoint:
            agent.learn(observation, action, reward, observation_, done)
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()

    Succ_Steps=list(env.state).count(0)
    succ_history.append(Succ_Steps)
    if Succ_Steps== Remove_num:
        agent.save_models()
        print('Solved after %.0f episodes.'% i)
        i=n_games+2

    avg_succ = np.mean(succ_history[-100:])
    Percent_Succ=Succ_Steps/Remove_num
    per_history.append(Percent_Succ)
    avg_percent=np.mean(per_history[-100:])
    toc=time.perf_counter()
    TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
    'Percent Successful':Percent_Succ,'Time':round((toc-tic),3)}, ignore_index=True)
    print('Episode ', i, '  Score %.1f' % score,'  Succesful Steps %.0f'% Succ_Steps,'  Avg_score %.1f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percentage %.0f' %(avg_percent*100),'  Time (s) %.0f' %(toc-tic))
    if i%5000==0:
        TrialData.to_pickle('TopOpt_6b6_V_V2_R2_TrialData.pkl')
    if not load_checkpoint and i%10000==0 and i>0 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)

    if not load_checkpoint and i==n_games-1 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)

    if avg_percent==.98:
        i=n_games+2

