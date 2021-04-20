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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import time
from gym import Env
from gym.spaces import Discrete, Box

import FEA_SOLVER_V
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
Max_SE_Tot=FEA_SOLVER_V.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,Stress=True)[1]
#---Loading Boundary Conditions----

LeftBC=list(range(0,Elements_X*(Elements_Y-1)*2,Elements_X*(Elements_Y-1)))
LeftBC.append(Elements_X-1) #For Bottom Right
class TopOpt_6by6_V(Env):
    def __init__(self):
        #Actons we can take... remove any of the blocks
        self.action_space=Discrete(ElementSize)
        #Current Material Matrix
        Low=np.array([0]*NodeSize)
        High=np.array([1]*NodeSize)
        self.observation_space=Box(Low,High,dtype=np.float32)
        
        # Set Maximum Number of Blocks to Remove
        self.Remove_Tot=Remove_num
        
    def step(self,action, New,preset_reward,preset_state):
        #Apply Action
  
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
            
                Run_Results=FEA_SOLVER_V.FEASolve(list(self.VoidCheck),Lx,Ly,Elements_X,Elements_Y,Stress=True)
        
                Max_SE_Ep=Run_Results[1]
                self.state=Run_Results[3]
                reward=(Max_SE_Tot/Max_SE_Ep)*It
            else:
                reward=preset_reward
                self.state=[preset_state]
    
        else:
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-10)"""
            reward=-1
            self.state=np.zeros((1,NodeSize))[0]
            done=True
            
        #PLaceholder for Info
        info={}
    
        return np.array(self.state), reward, done, info
    
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
env.Test=False 
env.Loc_List=[]
env.Dir_List=[]
env.Rem_List=[]
      
class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
            name='actor_critic_6b6_V_V8', chkpt_dir='tmp/actor_critic'):
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
    def __init__(self, n_actions,alpha, gamma=0.99, ):
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

        #log_prob = action_probabilities.log_prob(action)
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
    
def plot_success_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running Success % of previous 100 scores')
    plt.xlabel('Episodes')
    plt.ylabel(' Average Success %')
    plt.savefig(figure_file)
        
#env = gym.make('LunarLander-v2')
#env = gym.make('CartPole-v0')
agent = Agent(n_actions=env.action_space.n, alpha=1e-5, )
n_games = 100_000
# uncomment this line and do a mkdir tmp && mkdir video if you want to
# record video of the agent playing the game.
#env = wrappers.Monitor(env, 'tmp/video', video_callable=lambda episode_id: True, force=True)
filename = 'TopOpt_6by6_V_V8'

figure_file = 'plots/' + filename +'_reward.png'
Success_figure='plots/' +filename +'_success.png'

best_score = env.reward_range[0]
score_history = []
step_history=[]
per_history=[]
succ_history=[]

load_checkpoint = False 
Retrain=False
if load_checkpoint:
    agent.load_models()
if Retrain is False:
    TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Time'])
if Retrain is True:
    agent.load_models()
    TrialData=pd.read_pickle(filename+'_TrialData.pkl')
Start_State=FEA_SOLVER_V.FEASolve(list(np.ones((1,ElementSize))[0]),Lx,Ly,Elements_X,Elements_Y,Stress=True)[3]
Observation_History=np.ones((1,NodeSize))
Observation_History=np.append(Observation_History,0)
Observation_History=np.vstack((Observation_History,Observation_History))
Reward_History=[-1,-1]
Next_Observation_History=Start_State
Next_Observation_History=np.vstack((Next_Observation_History,Start_State))
for i in range(n_games):
    observation = env.reset(Start_State)
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        Data_Check=np.append(observation,action)
        Data_Loc=np.where((Observation_History==Data_Check).all(axis=1))
                
        if len(Data_Loc[0])==0:
            preset_reward=0
            preset_state=0
            New=True
        else:
            preset_state=Next_Observation_History[Data_Loc[0][0]]
            preset_reward=Reward_History[Data_Loc[0][0]]
            New=False
        observation_, reward, done, info = env.step(action,New,preset_reward,preset_state)
        Reward_History=np.append(Reward_History,reward)
        Next_Observation_History=np.vstack((Next_Observation_History,observation_))
        Observation_History=np.vstack((Observation_History,Data_Check))
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

    Succ_Steps=list(env.VoidCheck).count(0)
    succ_history.append(Succ_Steps)


    avg_succ = np.mean(succ_history[-100:])
    Percent_Succ=Succ_Steps/Remove_num
    per_history.append(Percent_Succ)
    avg_percent=np.mean(per_history[-100:])
    toc=time.perf_counter()
    TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
    'Percent Successful':Percent_Succ,'Time':round((toc-tic),3)}, ignore_index=True)
    print('Episode ', i, '  Score %.1f' % score,'  Succesful Steps %.0f'% Succ_Steps,'  Avg_score %.1f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percentage %.0f' %(avg_percent*100),'  Time (s) %.0f' %(toc-tic))
    if i%1000==0:
        TrialData.to_pickle(filename +'_TrialData.pkl')
    if not load_checkpoint and i%1000==0 and i>0 :
        x = range(0,i+1)
        plt.figure()
        plot_learning_curve(x, score_history, figure_file)
        plot_success_curve(x, per_history, Success_figure)
    if not load_checkpoint and i==n_games-1 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)
        plot_success_curve(x, per_history, Success_figure)
    if avg_percent>.99:
        print('Solved in '+str(i)+'iterations.')
        x = range(0,i+1)
        plt.figure()
        plot_learning_curve(x, score_history, figure_file)
        plot_success_curve(x, per_history, Success_figure)
        TrialData.to_pickle(filename+'_TrialData.pkl')
        agent.save_models()
        break 

