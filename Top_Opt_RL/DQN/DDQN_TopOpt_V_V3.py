# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""


''' Nathan Brown 
Policy Gradient Training of Topology Optimization through Reinforcement learning'''

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import os
import random
import tensorflow.keras as keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation

import pandas as pd
import matplotlib.pyplot as plt
import time
from gym import Env
from gym.spaces import Discrete, Box
import scipy
import FEA_SOLVER_V
import itertools
#Define The Size and Scope of Your Training
tic=time.perf_counter()

Lx=1
Ly=1
Elements_X=6
Elements_Y=6
ElementSize=Elements_X*Elements_Y
NodeSize=(Elements_X+1)*(Elements_Y+1)
Vol_fraction=16/36
Remove_num=ElementSize-(ElementSize*Vol_fraction)
VoidCheck=np.ones((1,ElementSize))
VoidCheck=list(VoidCheck[0])
#---Loading Boundary Conditions----
Max_SE_Tot=np.max((FEA_SOLVER_V.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,Stress=True)[1]))
x=np.array([int(Remove_num),0,0,int(Remove_num)])
y=np.array([0,0,0.8,0.8])
z=np.array([0.1,0,0.5,1])

def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G
ordr=2
G = poly_matrix(x, y, ordr)
# Solve for np.dot(G, m) = z:
m = np.linalg.lstsq(G, z)[0]
nx, ny = int(Remove_num), 100
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                     np.linspace(y.min(), y.max(), ny))
GoG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
zz = np.reshape(np.dot(GoG, m), xx.shape)
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
        
        # Set Maximum Number of Blocks to Remove
        self.Remove_Tot=Remove_num
        
    def step(self,action, New,preset_reward,preset_state,preset_SE):
        #Apply Action

            
        # evaluate it on grid
        rs_place=self.VoidCheck[action]
        self.VoidCheck[action]=0
        ElementMat=np.reshape(self.VoidCheck,(Elements_X,Elements_Y))
        SingleCheck=FEA_SOLVER_V.isolate_largest_group_original(ElementMat)
        It=list(self.VoidCheck).count(0)
        if rs_place==1 and action not in LeftBC and SingleCheck[1]==True:
            if It>=self.Remove_Tot:
                done=True 
            else:
                done=False

            if New is True:
                print('New')
                Run_Results=FEA_SOLVER_V.FEASolve(list(self.VoidCheck),Lx,Ly,Elements_X,Elements_Y,Stress=True)
        
                Max_SE_Ep=np.max(Run_Results[1])

                reward = zz[(int((Max_SE_Tot/Max_SE_Ep)*100))-1,It-1]
                self.state=Run_Results[3]
                
            else:
                print('Old')
                reward = zz[(int((Max_SE_Tot/preset_SE)*100))-1,It-1]

                self.state=[preset_state]
                Max_SE_Ep=preset_SE
            
    
        else:
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-10)"""

            reward=-1
            done=True
            Max_SE_Ep=0
            
        #PLaceholder for Info
        info={}
    
        return np.array(self.state), reward, done, It, Max_SE_Ep,info
    
    def render(self, mode='human'):
        ElementMat=np.reshape(self.VoidCheck,(Elements_X,Elements_Y))
        print(ElementMat)
        print('')
        
    def reset(self,Start_State):
        self.VoidCheck=np.ones((1,ElementSize))
        VoidCheck=np.ones((1,ElementSize))
        VoidCheck=list(VoidCheck[0])
        self.VoidCheck = VoidCheck
        self.VoidCheck=np.array(self.VoidCheck)
        self.state = Start_State
        self.state=np.array(self.state)
        return self.state
    
env = TopOpt_6by6_V()

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims, fc2_dims):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = Dense(fc1_dims, activation='relu')
        self.dense2 = Dense(fc2_dims, activation='relu')
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A

class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims,filename, epsilon_dec=5e-5, eps_end=0.01, 
                 mem_size=50000, fc1_dims=128,
                 fc2_dims=128, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.q_next = DuelingDeepQNetwork(n_actions, fc1_dims, fc2_dims)
        self.checkpoint_file=filename+'_q_weights'
        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation,load_checkpoint):
        if np.random.random() < self.epsilon and load_checkpoint is False:
            action = np.random.choice(self.action_space)
        else:
            state = np.array(observation)
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            Loss=.5
            return Loss

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)
        # changing q_pred doesn't matter because we are passing states to the train function anyway
        # also, no obvious way to copy tensors in tf2?
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        
        Loss=np.subtract(q_target,q_pred.numpy())
        Loss=np.square(Loss)
        Loss=Loss.mean()
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min

        self.learn_step_counter += 1
        return Loss
        
    def save_models(self):
        print('... saving models ...')
        self.q_eval.save_weights(self.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.q_eval.load_weights(self.checkpoint_file)
        
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
filename = 'DDQN_TopOpt_6by6_V_V3'
agent = Agent(lr=0.0005, gamma=0.99,filename=filename, n_actions=env.action_space.n, epsilon=1.0,
                  batch_size=128, input_dims=[env.observation_space.shape[0]])
n_games = 50_000
# uncomment this line and do a mkdir tmp && mkdir video if you want to
# record video of the agent playing the game.


figure_file = 'plots/' + filename +'_reward.png'
Success_figure='plots/' +filename +'_success.png'

best_score = env.reward_range[0]
score_history = []
step_history=[]
per_history=[]
succ_history=[]
Loss_history=[]
Ep_History=[]

load_checkpoint = False
Retrain=False
if load_checkpoint:
    agent.load_models()
    n_games=1
if not load_checkpoint:
    TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Avg Loss','Epsilon','Time'])
    MappingData=pd.DataFrame(columns=['Episode','Reward','Action','Blocks Removed'])

Start_State=FEA_SOLVER_V.FEASolve(list(np.ones((1,ElementSize))[0]),Lx,Ly,Elements_X,Elements_Y,Stress=True)[3]
Ep_History=[1,1]
SE_History=[0,0]
Observation_History=np.ones((1,ElementSize))
Observation_History=np.append(Observation_History,0)
Observation_History=np.vstack((Observation_History,Observation_History))
Reward_History=[-1,-1]
Next_Observation_History=Start_State
Next_Observation_History=np.vstack((Next_Observation_History,Start_State))
if not load_checkpoint:
    MappingData=MappingData.append({'Episode': 0, 'Reward': 0, 'Action': 0,'Blocks Removed': 0},ignore_index=True)
for i in range(n_games):
    observation = env.reset(Start_State)
    done = False
    score = 0

    while not done:
        action = agent.choose_action(observation,load_checkpoint)
        Data_Check=np.append(observation,action)
        tic2=time.perf_counter()

        Data_Loc=np.where((Observation_History==Data_Check).all(axis=1))
        if len(Data_Loc[0])==0:
            preset_reward=0
            preset_state=0
            preset_SE=0
            New=True
        else:
            
            preset_SE=SE_History[Data_Loc[0][0]]
            preset_state=Next_Observation_History[Data_Loc[0][0]]
            preset_reward=Reward_History[Data_Loc[0][0]]
            New=False
        observation_, reward, done, It, SE,info = env.step(action,New,preset_reward,preset_state,preset_SE)
        toc2=time.perf_counter()
        print(round(toc2-tic2,4))
        if i%50==0:
            env.render()
        agent.store_transition(observation,action,reward,observation_,done)
        if New is True:
            Reward_History=np.append(Reward_History,reward)
            Next_Observation_History=np.vstack((Next_Observation_History,observation_))
            Observation_History=np.vstack((Observation_History,Data_Check))
            
            SE_History=np.append(SE_History,SE)
        score += reward
        if not load_checkpoint:
            MappingData=MappingData.append({'Episode': i, 'Reward': reward, 'Action': action,'Blocks Removed': It},ignore_index=True)
        if load_checkpoint:
            env.render()
        if not load_checkpoint:
            Total_Loss=agent.learn()
            Loss_history.append(Total_Loss)
            avg_Loss=np.mean(Loss_history[-100:])
        observation = observation_
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    max_score=np.max(score_history)
    

    if avg_score > best_score and not load_checkpoint:
        best_score = avg_score
        if not load_checkpoint:
            agent.save_models()

    Succ_Steps=list(env.VoidCheck).count(0)
    succ_history.append(Succ_Steps)


    avg_succ = np.mean(succ_history[-100:])
    Percent_Succ=Succ_Steps/Remove_num
    per_history.append(Percent_Succ)
    avg_percent=np.mean(per_history[-100:])
    toc=time.perf_counter()
    if not load_checkpoint:
        TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
                'Percent Successful':Percent_Succ,'Avg Loss':avg_Loss,'Epsilon': agent.epsilon, 'Time':round((toc-tic),3)}, ignore_index=True)
    print('Episode ', i, '  Score %.2f' % score,'  Avg_score %.2f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percent %.0f' %(avg_percent*100),' Avg. Loss %.5f' %avg_Loss,'  Ep.  %.2f' %agent.epsilon,'  Time (s) %.0f' %(toc-tic))
    if i%100==0:
        TrialData.to_pickle(filename +'_TrialData.pkl')
        MappingData.to_pickle(filename+'_MappingData.pkl')
    if not load_checkpoint and i%100==0 and i>0 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)
 
    if not load_checkpoint and i==n_games-1 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)
  

    if avg_percent>.99 and abs(avg_Loss)<1e-4:
        print('Solved in '+str(i)+'iterations.')
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)
        MappingData.to_pickle(filename+'_MappingData.pkl')
        TrialData.to_pickle(filename+'_TrialData.pkl')
        break 

