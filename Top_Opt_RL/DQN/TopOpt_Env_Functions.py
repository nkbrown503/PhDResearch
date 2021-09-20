# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:00:02 2021

@author: nbrow
"""
from gym import Env
from gym.spaces import Discrete
import numpy as np
import pandas as pd
import itertools
from Node_Element_Extraction import BC_Nodes,LC_Nodes,Element_Lists
import FEA_SOLVER_GENERAL
import math
import copy
import json
from Matrix_Transforms import Condition_Transform
import matplotlib.pyplot as plt
import random
import sys

class TopOpt_Gen(Env):
    def __init__(self,Elements_X,Elements_Y,Vol_Frac,SC,opts):
        #Actons we can take... remove any of the blocks
        self.EX=Elements_X
        self.p=opts.P_Norm
        self.RS=Reward_Surface(opts)[0]
        self.RV=Reward_Surface(opts)[1]
        self.SC=SC
        self.Lx=opts.Lx
        self.EY=Elements_Y
        self.Ly=opts.Ly
        self.action_space=Discrete(self.EX*self.EY)
        self.eta=opts.Eta
        self.Vol_Frac=Vol_Frac
    def step(self,action,observation,Last_Reward,load_checkpoint,env,PR,FEA_Skip):
        #Apply Action
        self.Counter+=1    
        # evaluate it on grid

        rs_place=self.VoidCheck[int(action)]
        self.VoidCheck[int(action)]=0
        ElementMat=np.reshape(self.VoidCheck,(self.EX,self.EY))
        SingleCheck=FEA_SOLVER_GENERAL.isolate_largest_group_original(ElementMat)
        It=list(self.VoidCheck).count(0)
        if rs_place==1 and action not in self.BC and SingleCheck[1]==True:
            done=False
            if It>=math.ceil((self.EX*self.EY)*(1-self.Vol_Frac)) and load_checkpoint or It>=math.ceil((self.EX*self.EY)*(1-self.Vol_Frac)) and PR:
                done=True
            if self.Counter==1 or (self.Counter/FEA_Skip)==int(self.Counter/FEA_Skip):
                Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),self.Lx,self.Ly,self.EX,self.EY,self.LC_Nodes,self.Load_Directions,self.BC_Nodes,Stress=True)
                self.Max_SE_Ep=np.max(Run_Results[1])
                if (env.P_Norm/(sum(sum([number**self.p for number in np.reshape(Run_Results[2],(1,self.EX*self.EY))]))**(1/self.p)))<(1-float(self.SC)):

                    done=True
                    print('STRESS CONSTRAINT HIT!')
            else:
                self.Stress_state=np.reshape(self.Stress_state,(1,self.EX*self.EY))
                self.Stress_state[0][action]=0
                self.Stress_state=np.reshape(self.Stress_state,(self.EX,self.EY))
            
            if abs(self.Max_SE_Tot/self.Max_SE_Ep)>=1 or abs(It/(self.EX*self.EY))>=1 or self.Max_SE_Tot==0 or self.Max_SE_Ep==0:
                reward=-1
                done=True
            else:
                reward = self.RS[(int((self.Max_SE_Tot/self.Max_SE_Ep)*1000))-1,int((It/(self.EX*self.EY))*1000)-1]
                if not load_checkpoint:
                    reward2=self.RV[int(1-(np.reshape(self.Stress_state,(self.EX*self.EY,1))[action])*1000)-1]
                    reward=np.mean([reward,reward2])
            if self.Counter==1 or (self.Counter/FEA_Skip)==int(self.Counter/FEA_Skip):
                self.Stress_state=Run_Results[3]
                self.Stress_state=np.reshape(self.Stress_state,(self.EX,self.EY))
            self.state=np.zeros((self.EX,self.EY,3))
            self.state[:,:,0]=self.Stress_state
            self.state[:,:,1]=self.BC_state
            self.state[:,:,2]=self.LC_state
        else:
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-1)"""
            Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),self.Lx,self.Ly,self.EX,self.EY,self.LC_Nodes,self.Load_Directions,self.BC_Nodes,Stress=True)
            self.Max_SE_Ep=np.max(Run_Results[1])
            self.Stress_state=Run_Results[3]
            self.Stress_state=np.reshape(self.Stress_state,(self.EX,self.EY))
            self.state=np.zeros((self.EX,self.EY,3))
            self.state[:,:,0]=self.Stress_state
            self.state[:,:,1]=self.BC_state
            self.state[:,:,2]=self.LC_state
            reward=-1
            done=True
            if rs_place==1:
                self.VoidCheck[int(action)]=1
            
        reward+=1e-4
        Last_Reward+=1e-4
        rho=((reward)-(Last_Reward))/min([reward,Last_Reward])
        if reward>Last_Reward:
            llambda=1
        else:
            llambda=-1
        x=rho+llambda
        f_x=math.atan(x*(math.pi/2)*(1/self.eta))
        reward=reward+(f_x-llambda)*abs(reward)
        

        return self.state, reward, done, It
    
    def render(self,mode='human'):
        RenderMat=copy.deepcopy(self.VoidCheck)
        for RM in range(0,len(self.BC_Elements)):
            RenderMat[int(self.BC_Elements[RM])]=2
            RenderMat[int(self.BC_Elements[RM])]=2
        for RM in range(0,len(self.LC_Elements)):
            RenderMat[int(self.LC_Elements[RM])]=4
        RenderMat=np.reshape(RenderMat,(self.EX,self.EY))
        print(np.flip(RenderMat,0))
        print('')
        return 
        
    def reset(self):

        self.Results=FEA_SOLVER_GENERAL.FEASolve(self.VoidCheck,self.Lx,self.Ly,self.EX,self.EY,self.LC_Nodes,self.Load_Directions,self.BC_Nodes,Stress=True)
        self.Stress_state=self.Results[3]
        self.P_Norm=sum(sum([number**self.p for number in np.reshape(self.Results[2],(1,self.EX*self.EY))]))**(1/self.p)        #self.Stress_state=list(np.array(self.Stress_state)
        self.Stress_state=np.reshape(self.Stress_state,(self.EX,self.EY))
        self.state=np.zeros((self.EX,self.EY,3))
        self.state[:,:,0]=self.Stress_state
        self.state[:,:,1]=self.BC_state
        self.state[:,:,2]=self.LC_state
        self.Counter=0

        return self.state
    def reset_conditions(self):
        self.Max_SE_Tot=0
        self.VoidCheck=np.ones((1,self.EX*self.EY))
        self.VoidCheck=list(self.VoidCheck[0])
        self.VoidCheck=np.array(self.VoidCheck)

        while self.Max_SE_Tot<=0 or self.Max_SE_Tot>5000:        
            self.BC_Elements=[]
            self.BC_Nodes=[]
            self.LC_Elements=[]
            self.LC_Nodes=[]
            self.BC=[]
            self.Load_Types=[]
            self.Load_Directions=[]
            self.BC_Elements=np.append(self.BC_Elements,int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]])))
            self.BC_Elements=np.append(self.BC_Elements,int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]])))
            while self.BC_Elements[0]==self.BC_Elements[1]:
                self.BC_Elements[1]=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            self.BC_Nodes=np.append(self.BC_Nodes,BC_Nodes(int(self.BC_Elements[0]),self.Lx,self.Ly,self.EX,self.EY)[0])
            self.BC_Nodes=np.append(self.BC_Nodes,BC_Nodes(int(self.BC_Elements[0]),self.Lx,self.Ly,self.EX,self.EY)[1])
            self.BC_Nodes=np.append(self.BC_Nodes,BC_Nodes(int(self.BC_Elements[1]),self.Lx,self.Ly,self.EX,self.EY)[0])
            self.BC_Nodes=np.append(self.BC_Nodes,BC_Nodes(int(self.BC_Elements[1]),self.Lx,self.Ly,self.EX,self.EY)[1])

      
            self.LC_Elements=np.append(self.LC_Elements,int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]])))
            while self.LC_Elements[0] in self.BC_Elements:
                self.LC_Elements[0]=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            
            self.BC_set=np.append(self.BC_Elements,self.LC_Elements)
            self.LC_state=list(np.zeros((1,self.EX*self.EY))[0])
            for LCS in range(0,len(self.LC_Elements)):
                self.LC_state[int(self.LC_Elements[LCS])]=1
            self.LC_state=np.reshape(self.LC_state,(self.EX,self.EY))
            self.Load_Types=np.append(self.Load_Types,random.choice([0,1]))
            self.LC_Nodes=np.append(self.LC_Nodes,LC_Nodes(int(self.LC_Elements[0]),self.Load_Types,self.Load_Directions,self.Lx,self.Ly,self.EX,self.EY,LCS,Node_Location=False)[0])
            self.LC_Nodes=np.append(self.LC_Nodes,LC_Nodes(int(self.LC_Elements[0]),self.Load_Types,self.Load_Directions,self.Lx,self.Ly,self.EX,self.EY,LCS,Node_Location=False)[1])
            if self.Load_Types[0]==0: #Load will be applied vertically
                self.LC_Nodes[0]+=((self.EX+1)*(self.EY+1))
                self.LC_Nodes[1]+=((self.EX+1)*(self.EY+1))
            self.Load_Directions=np.append(self.Load_Directions,random.choice([-1,1])) #1 for Compressive Load, -1 for tensile load
            self.BC=np.append(self.BC,self.BC_Elements)
            self.BC=np.append(self.BC,self.LC_Elements)
            self.BC_state=list(np.zeros((1,self.EX*self.EY))[0])
            for BCS in range(0,len(self.BC_Elements)):
                self.BC_state[int(self.BC_Elements[BCS])]=1
            self.BC_state=np.reshape(self.BC_state,(self.EX,self.EY))
            self.Results=FEA_SOLVER_GENERAL.FEASolve(self.VoidCheck,self.Lx,self.Ly,self.EX,self.EY,self.LC_Nodes,self.Load_Directions,self.BC_Nodes,Stress=True)
            self.Max_SE_Tot=self.Results[1]
    def primer_cond(self,EX,EY):
         self.BC=[]
         self.BC=np.append(self.BC,self.BC_Elements)
         self.BC=np.append(self.BC,self.LC_Elements)
         self.BC_state=list(np.zeros((1,EX*EY))[0])
         for BCS in range(0,len(self.BC_Elements)):
                self.BC_state[int(self.BC_Elements[BCS])]=1
         self.BC_state=np.reshape(self.BC_state,(EX,EY))
         self.LC_state=list(np.zeros((1,EX*EY))[0])
         for LCS in range(0,len(self.LC_Elements)):
                self.LC_state[int(self.LC_Elements[LCS])]=1
         self.LC_state=np.reshape(self.LC_state,(EX,EY))
         self.Results=FEA_SOLVER_GENERAL.FEASolve(self.VoidCheck,self.Lx,self.Ly,self.EX,self.EY,self.LC_Nodes,self.Load_Directions,self.BC_Nodes,Stress=True)
         self.Max_SE_Tot=np.max(self.Results[1])

def Prog_Refine_Act(agent_primer,env,env_primer,load_checkpoint,Testing,opts,Small_EX,Small_EY,Time_Trial,From_App,FEA_Skip):
    '''This function will deliver the optimal topology of the smaller sized environment.
    This final topology will then be transformed into the equivalent topology at the 
    larger selected size. This larger topology will then be based back to the main function
    and the topology removal process will continue.'''
    Stable=False
    while not Stable:
        env_primer.BC_Nodes,env_primer.BC_Elements,env_primer.LC_Elements,env_primer.LC_Nodes,env_primer.Load_Directions=Condition_Transform(opts.Lx,opts.Ly,Small_EX,Small_EY,opts.Main_EX,opts.Main_EY,env.BC_Elements,env.LC_Elements,env.Load_Types,env.Load_Directions)
        
        LN_Hold=env_primer.LC_Nodes
        LT_Count=0
        for Counting in range(0,len(env_primer.LC_Nodes)):
            if Counting/2==int(Counting/2) and Counting/2!=0:
                LT_Count+=1
            if env.Load_Types[LT_Count]==0:
                env_primer.LC_Nodes[Counting]+=((Small_EX+1)*(Small_EY+1))
        env_primer.primer_cond(Small_EX,Small_EY)
        for Check in range(0,len(LN_Hold)):
            if LN_Hold[Check] in env_primer.BC_Nodes:
                if load_checkpoint:
                    print('-------------------------------')
                    print('Illegal Combination of BC and LC')
                    print('-------------------------------')
                    sys.exit()
                env.reset_conditions()
            else:
                Stable=True
    primer_done=False
    observation_primer=env_primer.reset()
    Last_Reward=0
    while not primer_done:
        Testing=True
        action = agent_primer.choose_action(observation_primer,load_checkpoint,Testing)
        observation_primer_, reward, primer_done, It = env_primer.step(action,observation_primer,Last_Reward,load_checkpoint,env,FEA_Skip=FEA_Skip,PR=True)
        observation_primer = observation_primer_
        Last_Reward=reward
        if load_checkpoint and not Time_Trial:
            env_primer.render()
    Last_Reward=0
    if Testing and not From_App:
        env_primer.render()
def App_Inputs(env,opts,User_Conditions):
    '''To improve the adaptability of this method, a web-app has been developed
    using Heroku The web-app will provide an interactive environment for the user
    to select the boundary and loading conditions. The BCs and LCs will be imported as 
    a .json file and distributed accordingly similar to the User_Input function'''
    env.BC_Elements=[]
    env.LC_Elements=[]
    env.BC_Nodes=[]
    env.LC_Nodes=[]
    env.Load_Types=[]
    env.Load_Directions=[]

    BC=[int(x)-1 for x in User_Conditions['bcs']]
    Right=[int(x)-1 for x in User_Conditions['rights']]
    Left=[int(x)-1 for x in User_Conditions['lefts']]
    Up=[int(x)-1 for x in User_Conditions['ups']]
    Down=[int(x)-1 for x in User_Conditions['downs']]
    env.BC_Elements=np.append(env.BC_Elements,BC)

    env.LC_Elements=np.append(env.LC_Elements,Right).astype('int')
    env.Load_Types=np.append(env.Load_Types,[1]*len(Right)).astype('int')
    env.Load_Directions=np.append(env.Load_Directions,[1]*len(Right)).astype('int')
    
    env.LC_Elements=np.append(env.LC_Elements,Left).astype('int')
    env.Load_Types=np.append(env.Load_Types,[1]*len(Left)).astype('int')
    env.Load_Directions=np.append(env.Load_Directions,[-1]*len(Left)).astype('int')
    
    env.LC_Elements=np.append(env.LC_Elements,Up).astype('int')
    env.Load_Types=np.append(env.Load_Types,[0]*len(Up)).astype('int')
    env.Load_Directions=np.append(env.Load_Directions,[1]*len(Up)).astype('int')
    
    env.LC_Elements=np.append(env.LC_Elements,Down).astype('int')
    env.Load_Types=np.append(env.Load_Types,[0]*len(Down)).astype('int')
    env.Load_Directions=np.append(env.Load_Directions,[-1]*len(Down)).astype('int')
    for Counting in range(0,len(env.LC_Elements)):
        if env.Load_Types[Counting]==0:
            LC_New_Nodes=LC_Nodes(int(env.LC_Elements[Counting]),env.Load_Types[Counting],env.Load_Directions[Counting],env.Lx,env.Ly,env.EX,env.EY,Counting,Node_Location=True)
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[0]+(opts.Main_EX+1)*(opts.Main_EY+1))
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[1]+(opts.Main_EX+1)*(opts.Main_EY+1))
        else:
            LC_New_Nodes=LC_Nodes(int(env.LC_Elements[Counting]),env.Load_Types[Counting],env.Load_Directions[Counting],env.Lx,env.Ly,env.EX,env.EY,Counting,Node_Location=True)
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[0])
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[1])
    for Counting in range(0,len(env.BC_Elements)):
        env.BC_Nodes=np.append(env.BC_Nodes,BC_Nodes(int(env.BC_Elements[Counting]),env.Lx,env.Ly,env.EX,env.EY)[0])
        env.BC_Nodes=np.append(env.BC_Nodes,BC_Nodes(int(env.BC_Elements[Counting]),env.Lx,env.Ly,env.EX,env.EY)[1])

    env.LC_state=list(np.zeros((1,(opts.Main_EX)*(opts.Main_EY)))[0])
    for LCS in range(0,len(env.LC_Elements)):
                env.LC_state[int(env.LC_Elements[LCS])]=1
    env.LC_state=np.reshape(env.LC_state,(opts.Main_EX,opts.Main_EY))
    env.BC=[]
    env.BC=np.append(env.BC,env.BC_Elements)
    env.BC=np.append(env.BC,env.LC_Elements)
    env.BC_state=list(np.zeros((1,(opts.Main_EX)*(opts.Main_EY)))[0])
    for BCS in range(0,len(env.BC_Elements)):
        env.BC_state[int(env.BC_Elements[BCS])]=1
    env.BC_state=np.reshape(env.BC_state,(opts.Main_EX,opts.Main_EY))
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)[1]))
    
    
def User_Inputs(env,opts):
    '''When testing a trained agent, the user will be prompted to select
    a single element to act as the loaded element, and two elements to act as the boundary 
    condition elements. Depending on where the elements are located, the nodes
    corresponding to these elements will be selected'''
    print(np.flip(np.reshape(range(0,(opts.Main_EX)*(opts.Main_EY)),(opts.Main_EX,opts.Main_EY)),0))
    BC_Count=int(input('How many boundary elements would you like to have: '))
    env.BC_Elements=[]
    env.LC_Elements=[]
    env.BC_Nodes=[]
    env.LC_Nodes=[]
    env.Load_Types=[]
    env.Load_Directions=[]
    for Counting in range(0,BC_Count):
        env.BC_Elements=np.append(env.BC_Elements,int(input('Please select an element to apply Boundary condition #'+str(Counting+1)+': ')))
        if env.BC_Elements[Counting]>(opts.Main_EX)*(opts.Main_EY) or env.BC_Elements[Counting]<0 or env.BC_Elements[Counting]!=int(env.BC_Elements[Counting]):
            print('Code Terminated By User...')
            sys.exit()
    print(np.flip(np.reshape(range(0,(opts.Main_EX*opts.Main_EY)),(opts.Main_EX,opts.Main_EY)),0))
    LC_Count=int(input('How many loading elements would you like to have: '))
    

    for Counting in range(0,LC_Count):
        env.LC_Elements=np.append(env.LC_Elements,int(input('Please select an element to apply the load #'+str(Counting+1)+': ')))
        if env.LC_Elements[Counting]>(opts.Main_EX)*(opts.Main_EY) or env.LC_Elements[Counting]<0 or env.LC_Elements[Counting]!=int(env.LC_Elements[Counting]):
            print('Code Terminated By User...')
            sys.exit()
        env.Load_Types=np.append(env.Load_Types,int(input('Input 0 for a Vertical Load or Input 1 for a Horizontal load for this element: ')))
        env.Load_Directions=np.append(env.Load_Directions,int(input('Input -1 for a tensile load or Input 1 for a compressive load for this element: ')))
    for Counting in range(0,LC_Count):
        if env.Load_Types[Counting]==0:
            LC_New_Nodes=LC_Nodes(int(env.LC_Elements[Counting]),env.Load_Types[Counting],env.Load_Directions[Counting],env.Lx,env.Ly,env.EX,env.EY,Counting,Node_Location=True)
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[0]+(opts.Main_EX+1)*(opts.Main_EY+1))
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[1]+(opts.Main_EX+1)*(opts.Main_EY+1))
        else:
            LC_New_Nodes=LC_Nodes(int(env.LC_Elements[Counting]),env.Load_Types[Counting],env.Lx,env.Ly,env.EX,env.EY,Counting,Node_Location=True)
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[0])
            env.LC_Nodes=np.append(env.LC_Nodes,LC_New_Nodes[1])
    for Counting in range(0,BC_Count):
        env.BC_Nodes=np.append(env.BC_Nodes,BC_Nodes(int(env.BC_Elements[Counting]),env.Lx,env.Ly,env.EX,env.EY)[0])
        env.BC_Nodes=np.append(env.BC_Nodes,BC_Nodes(int(env.BC_Elements[Counting]),env.Lx,env.Ly,env.EX,env.EY)[1])

    env.LC_state=list(np.zeros((1,(opts.Main_EX)*(opts.Main_EY)))[0])
    for LCS in range(0,len(env.LC_Elements)):
                env.LC_state[int(env.LC_Elements[LCS])]=1
    env.LC_state=np.reshape(env.LC_state,(opts.Main_EX,opts.Main_EY))
    env.BC=[]
    env.BC=np.append(env.BC,env.BC_Elements)
    env.BC=np.append(env.BC,env.LC_Elements)
    env.BC_state=list(np.zeros((1,(opts.Main_EX)*(opts.Main_EY)))[0])
    for BCS in range(0,len(env.BC_Elements)):
        env.BC_state[int(env.BC_Elements[BCS])]=1
    env.BC_state=np.reshape(env.BC_state,(opts.Main_EX,opts.Main_EY))
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)[1]))
    
    return 
def Testing_Inputs(env,opts):
    '''Every 200 episodes, the boundary and loading conditions
    should be set as those of a cantilever beam to monitor the progress
    of the agents learning'''
    env.BC_Nodes=np.array([0,0,opts.Main_EX*opts.Main_EY,opts.Main_EX*opts.Main_EY])
    env.LC_Nodes=np.array([opts.Main_EX+(opts.Main_EX+1)*(opts.Main_EY+1),opts.Main_EX-1+(opts.Main_EX)*(opts.Main_EY+1)])

    env.LC_Elements=np.array([np.where(FEA_SOLVER_GENERAL.rectangularmesh(opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY)[1]==env.LC_Nodes[0]-((opts.Main_EX+1)*(opts.Main_EY+1)))[0][0]])
    env.BC_Elements=[0,(opts.Main_EX)*(opts.Main_EY-1)]
    env.LC_state=list(np.zeros((1,(opts.Main_EX)*(opts.Main_EY)))[0])
    env.LC_state[env.LC_Elements[0]]=1
    env.LC_state=np.reshape(env.LC_state,(opts.Main_EX,opts.Main_EY))
    env.Load_Types=[0]
    env.Load_Directions=[-1] #1 for Compressive Load, -1 for tensile load
    env.BC=[]
    env.BC=np.append(env.BC,env.BC_Elements)
    env.BC=np.append(env.BC,env.LC_Elements)
    env.BC_state=list(np.zeros((1,(opts.Main_EX)*(opts.Main_EY)))[0])
    env.BC_state[env.BC_Elements[0]]=1
    env.BC_state[env.BC_Elements[1]]=1
    env.BC_state=np.reshape(env.BC_state,(opts.Main_EX,opts.Main_EY))
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)[1]))

def Testing_Info(env,env_primer,env_primer2,opts,score,Progressive_Refinement,From_App,Fixed):
    '''Function that outputs the results of a testing trial. The results include
    the score based on the reward function, the final strain energy, and if needed
    the number of arbitrary blocks removed by the shaving algorithm'''
    if not From_App:
        print('----------------')
    
        print('The final topology: ')
        for BC_Count in range(0,len(env.BC_Elements)):
            print('BC Element #'+str(BC_Count)+': '+str(int(env.BC_Elements[BC_Count])))
        for LC_Count in range(0,len(env.LC_Elements)):
            print('LC Element #'+str(LC_Count)+': '+str(int(env.LC_Elements[LC_Count])))
            if env.Load_Types[LC_Count]==0:
                Load_Types='Vertical'
            else:
                Load_Types='Horizontal'
            if env.Load_Directions[LC_Count]==-1:
                Load_Dir='Tensile'
            else:
                Load_Dir='Compressive'
            print('Load Type: '+Load_Dir)
            print('Load Direction: '+Load_Types)
      
        if Progressive_Refinement:
            env_primer.render()
            env_primer2.render()
        env.render()
        Final_Results=FEA_SOLVER_GENERAL.FEASolve(list(env.VoidCheck),opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)
        print('Strain Energy for Final Topology: '+str(round(np.max(Final_Results[1]),1)))
        p=opts.P_Norm
        print('Maximum P_Norm Stress Perc Increase: '+str(round(1-(env.P_Norm/sum(sum([number**p for number in np.reshape(Final_Results[2],(1,env.EX*env.EY))]))**(1/p)),2)))
        print('Final Volume Fraction: '+str(round(1-(list(env.VoidCheck).count(0)/(env.EX*env.EY)),3)))
        
        print('----------------')
        Mat_Plot=copy.deepcopy(env_primer.VoidCheck)
        plt.figure(1)
        for BC_Count in range(0,len(env_primer.BC_Elements)):
            Mat_Plot[int(env_primer.BC_Elements[BC_Count])]=3
        for LC_Count in range(0,len(env_primer.LC_Elements)):
            Mat_Plot[int(env_primer.LC_Elements[LC_Count])]=2
        plt.subplot(221)
        plt.imshow(np.flip(np.reshape(Mat_Plot,(opts.PR_EX,opts.PR_EY)),axis=0),cmap='Blues')
        Mat_Plot=copy.deepcopy(env_primer2.VoidCheck)
        for BC_Count in range(0,len(env_primer2.BC_Elements)):
            Mat_Plot[int(env_primer2.BC_Elements[BC_Count])]=3
        for LC_Count in range(0,len(env_primer2.LC_Elements)):
            Mat_Plot[int(env_primer2.LC_Elements[LC_Count])]=2
            plt.subplot(222)
        plt.imshow(np.flip(np.reshape(Mat_Plot,(opts.PR2_EX,opts.PR2_EY)),axis=0),cmap='Blues')
        Mat_Plot=copy.deepcopy(env.VoidCheck)
        for BC_Count in range(0,len(env.BC_Elements)):
            Mat_Plot[int(env.BC_Elements[BC_Count])]=3
        for LC_Count in range(0,len(env.LC_Elements)):
            Mat_Plot[int(env.LC_Elements[LC_Count])]=2
        plt.subplot(224)
        plt.imshow(np.flip(np.reshape(Mat_Plot,(opts.Main_EX,opts.Main_EY)),axis=0),cmap='Blues')
        plt.show()
    else:
        Final_Results=FEA_SOLVER_GENERAL.FEASolve(list(env.VoidCheck),opts.Lx,opts.Ly,opts.Main_EX,opts.Main_EY,env.LC_Nodes,env.Load_Directions,env.BC_Nodes,Stress=True)
        Mat_Plot=copy.deepcopy(env.VoidCheck)
        App_Plot={}
        App_Plot['Topology']=[]
        App_Plot['SE']=[]
        App_Plot['VF']=[]
        App_Plot['Topology'].append([str(x) for x in Mat_Plot])
        App_Plot['SE'].append(str(round(np.max(Final_Results[1]),1)))
        App_Plot['VF'].append(str(round(1-(list(env.VoidCheck).count(0)/(env.EX*env.EY)),3)))
        with open('Final_Top.txt', 'w') as outfile:
            json.dump(App_Plot,outfile)
        
def poly_matrix(x, y, order=2):
    """ Function to produce a matrix built on a quadratic surface """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G   

def Reward_Surface(opts):
    x=np.array([1,0,0,1,.5,0,.5])
    y=np.array([0,0,1,1,.5,.5,0])
    z=np.array([])
    a=opts.a
    b=opts.b
    for i in range(0,len(x)):
        z=np.append(z,(a*(x[i])**2)+(b*(y[i])**2))
    
    ordr=2
    G = poly_matrix(x, y, ordr)
    # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, z,rcond=None)[0]
    nx, ny = 1000, 1000
    
    xx, yy = np.meshgrid(np.linspace(0, 1, nx),
                         np.linspace(0, 1, ny))
    GoG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    zz = np.reshape(np.dot(GoG, m), xx.shape)
    Data=pd.read_pickle('Trial_Data/Reward_Data.pkl')
    Data=Data.to_numpy()
    X_Data=Data[:,0]
    Y_Data=Data[:,1]
    Z_Data=Data[:,2]

    GG = poly_matrix(X_Data, Y_Data, ordr)
# Solve for np.dot(G, m) = z:
    mm = np.linalg.lstsq(GG, Z_Data,rcond=None)[0]

    GoGG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    Reward_Ind = np.reshape(np.dot(GoGG, mm), xx.shape)[:,-1]
    return zz,Reward_Ind
