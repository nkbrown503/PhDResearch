"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""

''' Nathan Brown 
Policy Gradient Training of Topology Optimization through Reinforcement learning'''
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import json
from keract import get_activations, display_activations 
import FEA_SOLVER_GENERAL
from opts import parse_opts
from TopOpt_Env_Functions import TopOpt_Gen, Prog_Refine_Act,User_Inputs,App_Inputs, Testing_Inputs, Testing_Info     
from Matrix_Transforms import obs_flip, action_flip, Mesh_Triming, Mesh_Transform
from RL_Necessities import Agent 
def plot_learning_curve(x, scores, figure_file):
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

def TopOpt_Designing(Time_Trial,From_App,User_Conditions):
    if Progressive_Refinement:
        agent_primer= Agent(env_primer,mem_size=Mem_Size,epsilon_dec=Ep_decay,Increase=False,
                            lr=opts.LR, gamma=opts.Gamma,filename_save=filename_save+str(PR_EX)+'by'+str(PR_EY),
                            filename_load=filename_load+str(PR_EX)+'by'+str(PR_EY),EX=PR_EX,EY=PR_EY, n_actions=env.action_space.n, epsilon=0,
                            batch_size=opts.Batch_Size, input_dims=[PR_EX,PR_EY,3])
        
        agent_primer2= Agent(env_primer2,mem_size=Mem_Size,epsilon_dec=Ep_decay,Increase=False,
                            lr=opts.LR, gamma=opts.Gamma,filename_save=filename_save+str(PR2_EX)+'by'+str(PR2_EY),
                            filename_load=filename_load+str(PR_EX)+'by'+str(PR_EY),EX=PR2_EX,EY=PR2_EY, n_actions=env.action_space.n, epsilon=0,
                            batch_size=opts.Batch_Size, input_dims=[PR2_EX,PR2_EY,3])
        agent_primer.load_models()
        agent_primer2.load_models()
    
    agent = Agent(env,mem_size=Mem_Size,epsilon_dec=Ep_decay,Increase=False,
                  lr=opts.LR, gamma=opts.Gamma,filename_save=filename_save+str(Main_EX)+'by'+str(Main_EY),
                  filename_load=filename_load+str(PR_EX)+'by'+str(PR_EY),EX=Main_EX,EY=Main_EY, n_actions=env.action_space.n, epsilon=1.0,
                  batch_size=opts.Batch_Size, input_dims=[Main_EX,Main_EY,3])
    if load_checkpoint:
        agent.load_models()
    
    figure_file = 'plots/' + filename_save +'_reward.png'    
    best_score = env.reward_range[0]
    
    score_history = []
    step_history=[]
    per_history=[]
    succ_history=[]
    Loss_history=[]
    
    if not load_checkpoint:
        TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Avg Loss','SDEV','Epsilon','Time'])
    env.reset_conditions()
    if From_App:
        n_games=1
    for i in range(n_games):

        Testing = False #Used to render the environment and track learning of the agent 
        if load_checkpoint:
            'If the user wants to test the agent, the user will be prompted to input BC and LC elements'
            if From_App:
                App_Inputs(env,Lx,Ly,Main_EX,Main_EY,User_Conditions)
            else:
            
                User_Inputs(env,Lx,Ly,Main_EX,Main_EY)
        done = False
        score = 0
    
        if i%10==0 and i>=100:
            Testing=True
            if i%200==0:
                'Every 200 episodes, a special BC/LC will be used for monitoring purposes'
                Testing_Inputs(env,Lx,Ly,Main_EX,Main_EY)
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
            Prog_Refine_Act(agent_primer,env,env_primer,load_checkpoint,Testing,Lx,Ly,PR_EX,PR_EY,Main_EX,Main_EY,Time_Trial,From_App,FEA_Skip=1)
            #Progressive Refinement #2 Going for Intermediate to Final Mesh Size
            env_primer2.VoidCheck=Mesh_Transform(PR_EX,PR_EY,PR2_EX,PR2_EY,env_primer.VoidCheck)
            Prog_Refine_Act(agent_primer2,env,env_primer2,load_checkpoint,Testing,Lx,Ly,PR2_EX,PR2_EY,Main_EX,Main_EY,Time_Trial,From_App,FEA_Skip=1)
            #This outcome will now be used as the final mesh Size 
            env.VoidCheck=Mesh_Transform(PR2_EX,PR2_EY,Main_EX,Main_EY,env_primer2.VoidCheck)
            #Removed_Num=Mesh_Triming(env_primer,PR_EX,PR_EY)
            #Uncomment the above line if you want to incorporate mesh trimming

            observation[:,:,0]=np.reshape(FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,Lx,Ly,Main_EX,Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)[3],(Main_EX,Main_EY))
        observation_v, observation_h,observation_vh=obs_flip(observation,Main_EX,Main_EY)
        Last_Reward=0
        while not done:
            if i%1000==0 and i>=1: #Every 1000 iterations, show the activation maps
                activations = get_activations(agent.q_eval.model, observation.reshape(-1,Main_EX,Main_EY,3))
                display_activations(activations, save=False)
            action = agent.choose_action(observation,load_checkpoint,Testing)
            observation_, reward, done, It= env.step(action,observation,Last_Reward,load_checkpoint,env,FEA_Skip=3,PR=False)
            if not load_checkpoint:
                observation_v_,observation_h_,observation_vh_=obs_flip(observation_,Main_EX,Main_EY)
                action_v,action_h,action_vh=action_flip(action,Main_EX,Main_EY)
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
            Testing_Info(env,env_primer,env_primer2,Lx,Ly,Main_EX,Main_EY,PR_EX,PR_EY,PR2_EX,PR2_EY,score,Progressive_Refinement,From_App,Fixed=True)
        if not load_checkpoint:
            Total_Loss=agent.learn()
        else:
            Total_Loss=1
        score_history,per_history,succ_history,Loss_history,Succ_Steps,Percent_Succ,avg_succ,avg_score,avg_Loss,avg_percent=Data_History(score_history,per_history,succ_history,Loss_history,Total_Loss,score,Main_EX,Main_EY,i)
    
        if n_games!=1:
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
            TrialData.to_pickle('Trial_Data/'+filename_save +'_TrialData.pkl')
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
    Main_EX=opts.Main_EX
    Main_EY=opts.Main_EY
    PR_EX=opts.Sub_EX
    PR_EY=opts.Sub_EY
    PR2_EX=opts.Sub2_EX
    PR2_EY=opts.Sub2_EY
    Ep_decay=opts.Epsilon_Decay
    Mem_Size=opts.Memory_Size
    Lx=opts.Length_X
    Ly=opts.Length_Y
    n_games = opts.Num_Games
    filename_save = 'DDQN_TopOpt_Generalized_CNN_4L_'

    filename_load = 'DDQN_TopOpt_Generalized_CNN_4L_'
    
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
                VF3=0.25
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
    env = TopOpt_Gen(Lx,Ly,Main_EX,Main_EY,Vol_Frac_3,SC)
    env_primer= TopOpt_Gen(Lx,Ly,PR_EX,PR_EY,Vol_Frac_1,SC)
    env_primer2=TopOpt_Gen(Lx,Ly,PR2_EX,PR2_EY,Vol_Frac_2,SC)
    '------------------------------------------'
    TopOpt_Designing(Time_Trial,From_App,User_Conditions)
