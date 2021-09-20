# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""

''' Nathan Brown 
Policy Gradient Training of Topology Optimization through Reinforcement learning'''
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import time
import json
import FEA_SOLVER_GENERAL
from opts import parse_opts
from TopOpt_Env_Functions import TopOpt_Gen, Prog_Refine_Act,User_Inputs,App_Inputs, Testing_Inputs, Testing_Info     
from Matrix_Transforms import obs_flip, action_flip, Mesh_Transform
from RL_Necessities import Agent 
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

def TopOpt_Designing(Time_Trial,From_App,User_Conditions,opts,load_checkpoint):
    if Progressive_Refinement:
        agent_primer= Agent(env_primer,opts,Increase=False,filename_save=opts.filename_save+str(opts.PR_EX)+'by'+str(opts.PR_EY),
                            filename_load=opts.filename_load,EX=opts.PR_EX,EY=opts.PR_EY, n_actions=opts.PR_EX*opts.PR_EY,
                            epsilon=0,input_dims=[opts.PR_EX,opts.PR_EY,3])
                            
        agent_primer2= Agent(env_primer2,opts,Increase=False,filename_save=opts.filename_save+str(opts.PR2_EX)+'by'+str(opts.PR2_EY),
                            filename_load=opts.filename_load,EX=opts.PR2_EX,EY=opts.PR2_EY, n_actions=opts.PR2_EX*opts.PR2_EY, 
                            epsilon=0,input_dims=[opts.PR2_EX,opts.PR2_EY,3])
        agent_primer.load_models()
        agent_primer2.load_models()
    
    agent = Agent(env,opts,Increase=False,filename_save=opts.filename_save+str(opts.Main_EX)+'by'+str(opts.Main_EY),
                  filename_load=opts.filename_load,EX=opts.Main_EX,EY=opts.Main_EY, n_actions=opts.Main_EX*opts.Main_EY, 
                  epsilon=1.0, input_dims=[opts.Main_EX,opts.Main_EY,3])
    if load_checkpoint:
        agent.load_models()
    
    figure_file = 'plots/' + opts.filename_save +'_reward.png'    
    best_score = env.reward_range[0]
    
    score_history = []
    per_history=[]
    succ_history=[]
    Loss_history=[]
    
    if not load_checkpoint:
        import pandas as pd 
        TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Avg Loss','SDEV','Epsilon','Time'])
    env.reset_conditions()
    if From_App:
        opts.n_games=1
    for i in range(opts.n_games):

        Testing = False #Used to render the environment and track learning of the agent 
        if load_checkpoint:
            'If the user wants to test the agent, the user will be prompted to input BC and LC elements'
            if From_App:
                App_Inputs(env,opts,User_Conditions)
            else:
            
                User_Inputs(env,opts)
        done = False
        score = 0
    
        if i%10==0 and i>=100:
            Testing=True
            if i%200==0:
                'Every 200 episodes, a special BC/LC will be used for monitoring purposes'
                Testing_Inputs(env,opts)
                print('--------Testing Run------')
        env.VoidCheck=list(np.ones((1,env.EX*env.EY))[0])
        if Time_Trial:
            Start_Time_Trial=time.perf_counter()
        observation = env.reset()
        print(env)
        if Progressive_Refinement:
            ''' Set Up to Complete 3 Iterations of Progressive Refinement'''
            #Progressive Refinement #1 Going from Smallest to Intermediate Mesh Size
            env_primer.VoidCheck=list(np.ones((1,env_primer.EX*env_primer.EY))[0])
            Prog_Refine_Act(agent_primer,env,env_primer,load_checkpoint,Testing,opts,opts.PR_EX,opts.PR_EY,Time_Trial,From_App,FEA_Skip=1)
            #Progressive Refinement #2 Going for Intermediate to Final Mesh Size
            env_primer2.VoidCheck=Mesh_Transform(opts.PR_EX,opts.PR_EY,opts.PR2_EX,opts.PR2_EY,env_primer.VoidCheck)
            Prog_Refine_Act(agent_primer2,env,env_primer2,load_checkpoint,Testing,opts,opts.PR2_EX,opts.PR2_EY,Time_Trial,From_App,FEA_Skip=1)
            #This outcome will now be used as the final mesh Size 
            env.VoidCheck=Mesh_Transform(opts.PR2_EX,opts.PR2_EY,opts.Main_EX,opts.Main_EY,env_primer2.VoidCheck)
            #Removed_Num=Mesh_Triming(env_primer,PR_EX,PR_EY)
            #Uncomment the above line if you want to incorporate mesh trimming

            observation[:,:,0]=np.reshape(FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)[3],(opts.Main_EX,opts.Main_EY))
        observation_v, observation_h,observation_vh=obs_flip(observation,opts.Main_EX,opts.Main_EY)
        Last_Reward=0
        while not done:
            if i%1000==0 and i>=1: #Every 1000 iterations, show the activation maps
                from keract import get_activations, display_activations 
                activations = get_activations(agent.q_eval.model, observation.reshape(-1,opts.Main_EX,opts.Main_EY,3))
                display_activations(activations, save=False)
            action = agent.choose_action(observation,load_checkpoint,Testing)
            observation_, reward, done, It= env.step(action,observation,Last_Reward,load_checkpoint,env,FEA_Skip=3,PR=False)
            if not load_checkpoint:
                observation_v_,observation_h_,observation_vh_=obs_flip(observation_,opts.Main_EX,opts.Main_EY)
                action_v,action_h,action_vh=action_flip(action,opts.Main_EX,opts.Main_EY)
                agent.store_transition(observation,action,reward,observation_,done)
                agent.store_transition(observation_v,action_v,reward,observation_v_,done)
                agent.store_transition(observation_h,action_h,reward,observation_h_,done)
                agent.store_transition(observation_vh,action_vh,reward,observation_vh_,done)
            score += reward
            Last_Reward=reward
            if Testing and not Time_Trial:
                env.render()
                print('Current Score: '+str(round(score,3)))
            observation = observation_
            if not load_checkpoint:
                observation_v=observation_v_
                observation_h=observation_h_
                observation_vh=observation_vh_
            if load_checkpoint and not Time_Trial:
                env.render()
        toc=time.perf_counter()

        if Time_Trial and not From_App:
            print('It took '+str(round(toc-Start_Time_Trial,1))+' seconds to complete this time trial.')
    
        if load_checkpoint:
            #Removed_Num=Mesh_Triming(env,Main_EX,Main_EY)   
            App_Plot=Testing_Info(env,env_primer,env_primer2,opts,score,Progressive_Refinement,From_App,Fixed=True)
        if not load_checkpoint:
            Total_Loss=agent.learn()
        else:
            Total_Loss=1
        score_history,per_history,succ_history,Loss_history,Succ_Steps,Percent_Succ,avg_succ,avg_score,avg_Loss,avg_percent=Data_History(score_history,per_history,succ_history,Loss_history,Total_Loss,score,opts.Main_EX,opts.Main_EY,i)
    
        if opts.n_games!=1:
            env.reset_conditions()
        if avg_score>=best_score and not load_checkpoint: 
            '''If the average score of the previous runs is better than 
            the previous best average then the new model should be saved'''
            agent.save_models()
            best_score=avg_score
    
        
        if not load_checkpoint:
            TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
                    'Percent Successful':Percent_Succ,'Avg Loss':avg_Loss,'Epsilon': agent.epsilon, 'Time':round((toc-tic),3)}, ignore_index=True)
        print('Episode ', i, '  Score %.2f' % score,'  Avg_score %.2f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percent %.0f' %(avg_percent*100),'     Avg Loss %.2f' %avg_Loss,'  Ep.  %.2f' %agent.epsilon,'  Time (s) %.0f' %(toc-tic))
        if i%100==0 and not load_checkpoint and i>0:
            TrialData.to_pickle('Trial_Data/'+opts.filename_save +'_TrialData.pkl')
            plot_learning_curve(range(0,i+1), score_history, figure_file)
     
tic=time.perf_counter()
if __name__=='__main__':
        #------------------------------------------
    # Information for App
    From_App=True
    
    File_Name='config.json'
    Json_File = open(File_Name) 
    User_Conditions = json.load(Json_File)
    #------------------------------------------
    
    'General Input' #Still need to adjust to account for parameter changes
    opts=parse_opts()

    
    '---------------------------------------'
    if From_App:
        LC=1
    else:
        LC=int(input('Would you like to train a new set of weights [0] or test a pretrained model [1]: '))
    
    if LC==0:
        load_checkpoint=False 
    else:
        load_checkpoint=True
    if load_checkpoint:
        if From_App:
            VF_S=0
        else:
            VF_S=int(input('Would you like to input a final volume fraction [0] or a final stress constraint [1]: '))
        if VF_S==0:
            if From_App:
                VF3=float(User_Conditions['volfraction'])
            else:
                VF3=float(input('Input a final volume fraction as a decimal (0,1): '))
            SC=10 #Ensure that the Stress Constraint is not triggered
        else:
            VF3=0
            SC=float(input('Input the acceptable percentage of stress increase that is acceptable as a decmial (0,1): '))
    if From_App:
        PR=1
    else:
        PR=int(input('Would like conduct design using progressive refinement? No [0]    Yes [1]: '))
    if PR==0:
        Progressive_Refinement=False 
    else:
        Progressive_Refinement=True
    if From_App:
        Time_Trial=True
    else:
        Time_Trial=False
    
    if load_checkpoint:
        Vol_Frac_3=VF3 
        if VF_S==0: #If the user wants to set a final volume fraction, set the intermediate volume fractions accordingly
            Vol_Frac_2=1-((1-Vol_Frac_3)/1.5)
            Vol_Frac_1=1-((1-Vol_Frac_3)/2.5)
        else:
            Vol_Frac_2=opts.Vol_Frac_2
            Vol_Frac_1=opts.Vol_Frac_1
    else:
        Vol_Frac_3=opts.Vol_Frac_3
        Vol_Frac_1=opts.Vol_Frac_1
        Vol_Frac_2=opts.Vol_Frac_2
        SC=10
    env = TopOpt_Gen(opts.Main_EX,opts.Main_EY,Vol_Frac_3,SC,opts)
    env_primer= TopOpt_Gen(opts.PR_EX,opts.PR_EY,Vol_Frac_1,SC,opts)
    env_primer2=TopOpt_Gen(opts.PR2_EX,opts.PR2_EY,Vol_Frac_2,SC,opts)
    '------------------------------------------'
    TopOpt_Designing(Time_Trial,From_App,User_Conditions,opts,load_checkpoint)
