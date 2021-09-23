# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""

''' Nathan Brown 
Main Function for the Reinforcement Learning Based Topology Optimization Solver using Double Q-Learning'''
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import time
import json
# import FEA_SOLVER_GENERAL
from Top_Opt_RL.DQN.FEA_SOLVER_GENERAL import *
# from Top_Opt_RL.DQN.FEA_SOLVER_GENERAL import FEA_SOLVER_GENERAL

from Top_Opt_RL.DQN.opts import parse_opts
from Top_Opt_RL.DQN.TopOpt_Env_Functions import TopOpt_Gen, Prog_Refine_Act,User_Inputs,App_Inputs, Testing_Inputs, Testing_Info     
from Top_Opt_RL.DQN.Matrix_Transforms import obs_flip, action_flip, Mesh_Transform
from Top_Opt_RL.DQN.RL_Necessities import Agent 
def plot_learning_curve(x, scores, figure_file):
    import matplotlib.pyplot as plt
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episodes')
    plt.ylabel(' Average Reward')
    plt.savefig(figure_file)
def Data_History(score_history,per_history,succ_history,Loss_history,Total_Loss,score,Main_EX,Main_EY,i):

    Loss_history.append(Total_Loss)
    avg_Loss=np.mean(Loss_history[-50:])
    score_history.append(score)
    avg_score = np.mean(score_history[-50:])
    Succ_Steps=list(env.VoidCheck).count(0)
    succ_history.append(Succ_Steps)

    avg_succ = np.mean(succ_history[-50:])
    Percent_Succ=Succ_Steps/(Main_EX*Main_EY)
    per_history.append(Percent_Succ)
    avg_percent=np.mean(per_history[-50:])
    return score_history,per_history,succ_history,Loss_history,Succ_Steps,Percent_Succ,avg_succ,avg_score,avg_Loss,avg_percent

def TopOpt_Designing(User_Conditions,opts, envs):
    Time_Trial = opts.Time_Trial
    if opts.Progressive_Refinement:
        agent_primer= Agent(envs.env_primer,opts,Increase=False,filename_save=opts.filename_save+str(opts.PR_EX)+'by'+str(opts.PR_EY),
                            filename_load=opts.filename_load,EX=opts.PR_EX,EY=opts.PR_EY, n_actions=opts.PR_EX*opts.PR_EY,
                            epsilon=0,input_dims=[opts.PR_EX,opts.PR_EY,3])
                            
        agent_primer2= Agent(envs.env_primer2,opts,Increase=False,filename_save=opts.filename_save+str(opts.PR2_EX)+'by'+str(opts.PR2_EY),
                            filename_load=opts.filename_load,EX=opts.PR2_EX,EY=opts.PR2_EY, n_actions=opts.PR2_EX*opts.PR2_EY, 
                            epsilon=0,input_dims=[opts.PR2_EX,opts.PR2_EY,3])
        agent_primer.load_models()
        agent_primer2.load_models()
    
    agent = Agent(envs.env,opts,Increase=False,filename_save=opts.filename_save+str(opts.Main_EX)+'by'+str(opts.Main_EY),
                  filename_load=opts.filename_load,EX=opts.Main_EX,EY=opts.Main_EY, n_actions=opts.Main_EX*opts.Main_EY, 
                  epsilon=1.0, input_dims=[opts.Main_EX,opts.Main_EY,3])
    if opts.Load_Checkpoints: agent.load_models()    
    figure_file = 'plots/' + opts.filename_save +'_reward.png'    
    best_score = envs.env.reward_range[0]    
    score_history ,per_history,succ_history,Loss_history= [],[],[],[]
    
    if not opts.Load_Checkpoints:
        from pandas import DataFrame 
        TrialData=DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Avg Loss','SDEV','Epsilon','Time'])
    envs.env.reset_conditions()
    if opts.From_App:  opts.n_games=1
    for i in range(opts.n_games):
        Testing = False #Used to render the environment and track learning of the agent 
        if opts.Load_Checkpoints:
            'If the user wants to test the agent, the user will be prompted to input BC and LC elements'
            if opts.From_App:  App_Inputs(envs.env,opts,User_Conditions)
            else:  User_Inputs(envs.env,opts)
        done = False
        score = 0    
        if i%10==0 and i>=100:
            Testing=True
            if i%200==0:
                'Every 200 episodes, a special BC/LC will be used for monitoring purposes'
                Testing_Inputs(envs.env,opts)
                print('--------Testing Run------')
        envs.env.VoidCheck=list(np.ones((1,envs.env.EX*envs.env.EY))[0])
        if Time_Trial:     Start_Time_Trial=time.perf_counter()
        observation = envs.env.reset()
        print(envs.env)
        if opts.Progressive_Refinement:
            ''' Set Up to Complete 3 Iterations of Progressive Refinement'''
            #Progressive Refinement #1 Going from Smallest to Intermediate Mesh Size
            envs.env_primer.VoidCheck=list(np.ones((1,envs.env_primer.EX*envs.env_primer.EY))[0])
            Prog_Refine_Act(agent_primer,envs.env,envs.env_primer,opts.Load_Checkpoints,Testing,opts,opts.PR_EX,opts.PR_EY,Time_Trial,opts.From_App,FEA_Skip=1)
            #Progressive Refinement #2 Going for Intermediate to Final Mesh Size
            envs.env_primer2.VoidCheck=Mesh_Transform(opts.PR_EX,opts.PR_EY,opts.PR2_EX,opts.PR2_EY,envs.env_primer.VoidCheck)
            Prog_Refine_Act(agent_primer2,envs.env,envs.env_primer2,opts.Load_Checkpoints,Testing,opts,opts.PR2_EX,opts.PR2_EY,Time_Trial,opts.From_App,FEA_Skip=1)
            #This outcome will now be used as the final mesh Size 
            envs.env.VoidCheck=Mesh_Transform(opts.PR2_EX,opts.PR2_EY,opts.Main_EX,opts.Main_EY,envs.env_primer2.VoidCheck)
            #Removed_Num=Mesh_Triming(env_primer,PR_EX,PR_EY)
            #Uncomment the above line if you want to incorporate mesh trimming

            observation[:,:,0]=np.reshape(FEASolve(envs.env.VoidCheck,opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,envs.env.LC_Nodes,envs.env.Load_Directions,envs.env.BC_Nodes,Stress=True)[3],(opts.Main_EX,opts.Main_EY))
        observation_v, observation_h,observation_vh=obs_flip(observation,opts.Main_EX,opts.Main_EY)
        Last_Reward=0
        while not done:
            if i%1000==0 and i>=1: #Every 1000 iterations, show the activation maps
                from keract import get_activations, display_activations 
                activations = get_activations(agent.q_eval.model, observation.reshape(-1,opts.Main_EX,opts.Main_EY,3))
                display_activations(activations, save=False)
            action = agent.choose_action(observation,opts.Load_Checkpoints,Testing)
            observation_, reward, done, It= envs.env.step(action,observation,Last_Reward,opts.Load_Checkpoints,envs.env,FEA_Skip=3,PR=False)
            if not opts.Load_Checkpoints:
                observation_v_,observation_h_,observation_vh_=obs_flip(observation_,opts.Main_EX,opts.Main_EY)
                action_v,action_h,action_vh=action_flip(action,opts.Main_EX,opts.Main_EY)
                agent.store_transition(observation,action,reward,observation_,done)
                agent.store_transition(observation_v,action_v,reward,observation_v_,done)
                agent.store_transition(observation_h,action_h,reward,observation_h_,done)
                agent.store_transition(observation_vh,action_vh,reward,observation_vh_,done)
            score += reward
            Last_Reward=reward
            if Testing and not Time_Trial:
                envs.env.render()
                print('Current Score: '+str(round(score,3)))
            observation = observation_
            if not opts.Load_Checkpoints:
                observation_v=observation_v_
                observation_h=observation_h_
                observation_vh=observation_vh_
            if opts.Load_Checkpoints and not Time_Trial:   envs.env.render()
        toc=time.perf_counter()

        if Time_Trial and not opts.From_App:
            print('It took '+str(round(toc-Start_Time_Trial,1))+' seconds to complete this time trial.')    
        if opts.Load_Checkpoints:
            #Removed_Num=Mesh_Triming(env,Main_EX,Main_EY)   
            App_Plot=Testing_Info(envs.env,envs.env_primer,envs.env_primer2,opts,score,opts.Progressive_Refinement,opts.From_App,Fixed=True)
            return App_Plot
        if not opts.Load_Checkpoints:Total_Loss=agent.learn()
        else:    Total_Loss=1
        score_history,per_history,succ_history,Loss_history,Succ_Steps,Percent_Succ,avg_succ,avg_score,avg_Loss,avg_percent=Data_History(score_history,per_history,succ_history,Loss_history,Total_Loss,score,opts.Main_EX,opts.Main_EY,i)
    
        if opts.n_games!=1: envs.env.reset_conditions()
        if avg_score>=best_score and not opts.Load_Checkpoints: 
            '''If the average score of the previous runs is better than 
            the previous best average then the new model should be saved'''
            agent.save_models()
            best_score=avg_score   
        
        if not opts.Load_Checkpoints:
            TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
                    'Percent Successful':Percent_Succ,'Avg Loss':avg_Loss,'Epsilon': agent.epsilon, 'Time':round((toc-tic),3)}, ignore_index=True)
        print('Episode ', i, '  Score %.2f' % score,'  Avg_score %.2f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percent %.0f' %(avg_percent*100),'     Avg Loss %.2f' %avg_Loss,'  Ep.  %.2f' %agent.epsilon,'  Time (s) %.0f' %(toc-tic))
        if i%100==0 and not opts.Load_Checkpoints and i>0:
            TrialData.to_pickle('Trial_Data/'+opts.filename_save +'_TrialData.pkl')
            plot_learning_curve(range(0,i+1), score_history, figure_file)
        # return App_Plot
     
class EnviromentsRL:
    def __init__(self, opts):
        if opts.Load_Checkpoints:
            Vol_Frac_3=opts.Vol_Frac_3 
            if opts.VF_S==0: #If the user wants to set a final volume fraction, set the intermediate volume fractions accordingly
                Vol_Frac_2=1-((1-Vol_Frac_3)/1.5)
                Vol_Frac_1=1-((1-Vol_Frac_3)/2.5)
            else:
                Vol_Frac_2=opts.Vol_Frac_2
                Vol_Frac_1=opts.Vol_Frac_1
        else:
            Vol_Frac_3=opts.Vol_Frac_3
        SC=opts.SC
        Vol_Frac_1=opts.Vol_Frac_1
        Vol_Frac_2=opts.Vol_Frac_2
        self.env = TopOpt_Gen(opts.Main_EX,opts.Main_EY,Vol_Frac_3,SC,opts)
        self.env_primer= TopOpt_Gen(opts.PR_EX,opts.PR_EY,Vol_Frac_1,SC,opts)
        self.env_primer2=TopOpt_Gen(opts.PR2_EX,opts.PR2_EY,Vol_Frac_2,SC,opts)


tic=time.perf_counter()
if __name__=='__main__':
    opts=parse_opts()   
    User_Conditions = json.load(open(opts.configfile) ) if opts.From_App else None  
    envs = EnviromentsRL(opts)  
    App_Plot=TopOpt_Designing(User_Conditions,opts, envs)
    json.dump( App_Plot, open( "App_Data.json", 'w' ) )
