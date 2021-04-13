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
import matplotlib.pyplot as plt
from collections import deque
import time
import pandas as pd
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
tic=time.perf_counter()
#---Loading State Table----------------
LeftBC=list(range(0,Elements_X*(Elements_Y-1)*2,Elements_X*(Elements_Y-1)))
LeftBC.append(Elements_X-1) #For Bottom Right
class TopOpt_6by6_V(Env):
    def __init__(self):
        #Actons we can take... remove any of the blocks
        self.action_space_row=Discrete(Elements_Y)
        self.action_space_col=Discrete(Elements_X)
        #Current Material Matrix
        Low=np.array([0]*(max(Elements_X,Elements_Y)))
        High=np.array([1]*(max(Elements_X,Elements_Y)))
        self.observation_space=Box(Low,High,dtype=np.float32)
        
        # Set Starting Material Matrix
        VoidCheck=np.ones((1,ElementSize))
        VoidCheck=list(VoidCheck[0])
        self.state = VoidCheck
        # Set Maximum Number of Blocks to Remove
        self.Remove_Tot=Remove_num
        
    def step(self,Row_R,Col_R):
        #Apply Action
        action_Loc=(Row_R*Elements_X)+(Col_R)
        rs_place=self.Mat_Rep[action_Loc]
        self.Mat_Rep[action_Loc]=0
        ElementMat=np.reshape(self.Mat_Rep,(Elements_X,Elements_Y))
        SingleCheck=FEA_SOLVER_V.isolate_largest_group_original(ElementMat)
        It=list(self.Mat_Rep).count(0)
        if rs_place==1 and action_Loc not in LeftBC and SingleCheck[1]==True:
            if It>=self.Remove_Tot:
                done=True 
            else:
                done=False
                
            reward=(FEA_SOLVER_V.FEASolve(list(self.Mat_Rep),Lx,Ly,Elements_X,Elements_Y)[1])*(-1*10**-3)
            reward=(1+reward[0][0])*It
            if reward<0:
                reward=-1
    
        else: 
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be punished (-1)"""
            reward=-1
            done=True
            
        #PLaceholder for Info
        info={}
        
        #Need to Count the Number of Solid Blocks in Each Row and Column 
        
        Ones_Count=np.where(ElementMat==1)
        self.state=[]
        for i in range(0,Elements_X):
            self.state.append(list(Ones_Count[0]).count(i))
            
        for i in range(0,Elements_Y):
            self.state.append(list(Ones_Count[1]).count(i))
            
        return np.array(self.state),np.array(self.Mat_Rep), reward, done, info
    
    def render(self, mode='human'):
        ElementMat=np.reshape(self.Mat_Rep,(Elements_X,Elements_Y))
        print(ElementMat)
        print('')
        
    def reset(self):
        VoidCheck=np.ones((1,ElementSize))
        VoidCheck=list(VoidCheck[0])
        self.Mat_Rep=VoidCheck
        state = np.ones(Elements_X)*Elements_Y
        state2=np.ones(Elements_Y)*Elements_X
        state_list=[list(state),list(state2)]
        
        self.state=sum(state_list,[])
        self.state=np.array(self.state)
        return self.state, self.Mat_Rep
    
env = TopOpt_6by6_V()
env.Test=False 
env.Loc_List=[]
env.Dir_List=[]
env.Rem_List=[]
      
class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions_row,n_actions_col, fc1_dims=512, fc2_dims=256,
            name='actor_critic_6b6_V_V3_R2', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions_row = n_actions_row
        self.n_actions_col = n_actions_col
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi_row = Dense(n_actions_row, activation='softmax')
        self.pi_col = Dense(n_actions_col, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi_row = self.pi_row(value)
        pi_col = self.pi_col(value)

        return v, pi_row, pi_col
    
class Agent:
    def __init__(self, n_actions_row,n_actions_col,alpha=0.001, gamma=0.999, ):
        self.gamma = gamma
        self.n_actions_row = n_actions_row
        self.n_actions_col = n_actions_col
        self.action_space_row = [i for i in range(self.n_actions_row)]
        self.action_space_col = [i for i in range(self.n_actions_col)]

        self.actor_critic = ActorCriticNetwork(n_actions_row=n_actions_row,n_actions_col=n_actions_col)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))


    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs_row,probs_col = self.actor_critic(state)

        action_probabilities_row = tfp.distributions.Categorical(probs=probs_row)
        action_probabilities_col = tfp.distributions.Categorical(probs=probs_col)
        
        Row_R = action_probabilities_row.sample()
        Col_R= action_probabilities_col.sample()
        log_prob_row = action_probabilities_row.log_prob(Row_R)
        log_prob_col = action_probabilities_col.log_prob(Col_R)
        self.Row_R = Row_R
        self.Col_R = Col_R
        return Row_R.numpy()[0], Col_R.numpy()[0]

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
            state_value, probs_row,probs_col = self.actor_critic(state)
            state_value_, _,_ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs_row = tfp.distributions.Categorical(probs=probs_row)
            log_prob_row = action_probs_row.log_prob(action[0])
            action_probs_col = tfp.distributions.Categorical(probs=probs_col)
            log_prob_col = action_probs_col.log_prob(action[1])
            
            delta = reward + self.gamma*state_value_*(1-int(done)) - state_value
            actor_loss_row = -log_prob_row*delta
            actor_loss_col = -log_prob_col*delta
            critic_loss = delta**2
            total_loss = actor_loss_row + actor_loss_col + critic_loss

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

env = TopOpt_6by6_V()
agent = Agent(alpha=1e-5, n_actions_row=env.action_space_row.n,n_actions_col=env.action_space_col.n)
n_games = 500_000
# uncomment this line and do a mkdir tmp && mkdir video if you want to
# record video of the agent playing the game.
#env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
filename = 'TopOpt_6by6_V_V3_R2.png'

figure_file = 'plots/' + filename

best_score = env.reward_range[0]
score_history = []
succ_history=[]
percent_history=[]
load_checkpoint = False
Retrain=True
if load_checkpoint:
    agent.load_models()
if Retrain is False:
    TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Time'])
if Retrain is True:
    agent.load_models()
    TrialData=pd.read_pickle('TopOpt_6b6_V_V3_R2_TrialData.pkl')
 
if load_checkpoint:
    agent.load_models()

for i in range(n_games):
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, Mat_Rep, reward, done, info = env.step(action[0],action[1])
        score += reward
        if not load_checkpoint:
            agent.learn(observation, action, reward, observation_, done)
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    Succ_Steps=list(Mat_Rep).count(0)
    succ_history.append(Succ_Steps)
    avg_succ = np.mean(succ_history[-100:])
    Percent_Succ=Succ_Steps/Remove_num
    percent_history.append(Percent_Succ)
    avg_percent=np.mean(percent_history[-100:])
    toc=time.perf_counter()
    TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
    'Percent Successful':Percent_Succ,'Time':round((toc-tic),3)}, ignore_index=True)
    if avg_percent>.98:
        i=n_games+2
    
    
    if avg_score > best_score:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()
    print('  Episode ', i, '  Score %.1f' % score, '  Avg. score %.1f' % avg_score,' Succesful Steps %.0f' % Succ_Steps, 'Avg. Steps %.0f' % avg_succ, 'Avg. Percentage %.0f' % (avg_percent*100),'Time (s): %.0f' % (toc-tic))
    if i%5000==0:
        TrialData.to_pickle('TopOpt_6b6_V_V3_R2_TrialData.pkl')
    if not load_checkpoint and i%10000==0 and i>0 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)
        
    if not load_checkpoint and i==n_games-1 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)

'''    
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
'''
