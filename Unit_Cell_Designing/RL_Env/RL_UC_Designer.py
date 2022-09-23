# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 13:18:38 2022

@author: nbrow
"""

import tensorflow as tf
#tf.compat.v1.enable_eager_execution()
import copy 
from UC_Env import UC_Env
import time
import matplotlib.pyplot as plt 
import gym
import argparse
import numpy as np
from DDPG import Agent
from utils import plot_learning_curve
import wandb
from wandb.keras import WandbCallback
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate",type=float,default=1e-4)
    parser.add_argument("--gamma",type=float,default=0.99)
    parser.add_argument("--max_noise",type=float,default=0.2)
    parser.add_argument("--noise_decay",type=float,default=5e-5)
    parser.add_argument("--Trial_Num",type=int,default=1)
    
    args=parser.parse_args()
    return args

args=parse_args()           
    
n_games = 10000 #Number of training Episodes

figure_file = 'plots/UC_RL_Training.png'

best_score = -10
score_history = []
Type='Tension'  #Change to either Tension or Compression depending on load type


Test=True #If Test=False then the RL agent will be trained from scratch
env = UC_Env(Type) #Call the RL environment 
agent = Agent(input_dims=(59,),Type=Type,alpha=5e-4,beta=5e-4,gamma=args.gamma,
              env=env,Start_Noise=args.max_noise,Noise_Decay=args.noise_decay,n_actions=7,TN=args.Trial_Num ) #Call the RL agent 

if Test==False:
    
    wandb.init(project='RL_UC_Training',
           config=vars(args),
           name='Trial_{}'.format(args.Trial_Num))
    for i in range(n_games):
        
        observation = env.reset(Test,Type,i=0)
        done = False
        evaluate=False
        score = 0
        steps=0
        print('here')
        while not done:
            steps+=1 
            action = agent.choose_action(observation, evaluate)

            observation_, reward, done, Legal = env.step(action)                
            if (np.round(observation_,3)==np.round(observation,3)).all()==True and done==False and Legal:
                #Check if the agent tried to take the same action back to back 
                done=True
                
                
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation=observation_
        S2=time.time()
        if agent.memory.mem_cntr>100:
            #Introduce a slight delay in the learning for the agent 
            agent.learn()


        

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score and i>20:
            best_score = avg_score
            Saved=False
            while Saved==False:
                try:
                    agent.save_models()
                    Saved=True
                except:
                    time.sleep(0.1)
        wandb.log({'reward': np.round(score,4),'avg reward': np.round(avg_score,4),'episode': i, 'noise': agent.noise})

        print('episode', i, 'score %.3f' % score, 'avg score %.3f' % avg_score, 'Steps %.0f' % steps, 'Noise %.4f' %agent.noise, ' LR %.3f' %agent.tau)

    
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
else:
    Testing_Trials=1 #How many trials would you like to test?
    n_steps = 0
    Avg_reward=[]
    MAE=[]
    agent.load_models()
    #fig2,ax2=plt.subplots()
    for i in range(Testing_Trials):
        
        observation=env.reset(Test,Type,i)
        done = False
        evaluate=True
        score = 0
        LR=-1
 
        while not done:
            action = agent.choose_action(observation, evaluate)

            observation_, reward, done, Legal = env.step(action)
            if reward>LR or reward==-1:
                score += reward
                print(reward)

                if reward!=0:
                    LR=reward
                    FE=env.Force_Error
                observation = observation_

            else:
                env.state_UC=env.state_UC_
                try:
                    env.Current_Force=env.Current_Force_
                except:
                    'Nothing'
                done=True
                
            if done:
                print(env.UC)
                env.render(Legal,i)
       
        if reward!=-1:
            MAE.append(FE)
            Avg_reward.append(LR)
        

        #print('Average Final Error: {}'.format(np.mean(Avg_reward)))
    