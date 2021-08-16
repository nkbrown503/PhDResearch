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
from opts import parse_opts
import statistics 
import math
import copy
from Matrix_Transforms import Condition_Transform

import random
import sys
from Matrix_Transforms import Mesh_Transform
opts=parse_opts()
class TopOpt_Gen(Env):
    def __init__(self,Lx,Ly,Elements_X,Elements_Y):
        #Actons we can take... remove any of the blocks
        self.EX=Elements_X
        self.RS=Reward_Surface()[0]
        self.RV=Reward_Surface()[1]
        
        self.Lx=Lx
        self.EY=Elements_Y
        self.Ly=Ly
        self.action_space=Discrete(self.EX*self.EY)
        self.eta=opts.Eta
    def step(self,action,observation,Last_Reward):
        #Apply Action
            
        # evaluate it on grid
      
        rs_place=self.VoidCheck[action]
        self.VoidCheck[action]=0
        ElementMat=np.reshape(self.VoidCheck,(self.EX,self.EY))
        SingleCheck=FEA_SOLVER_GENERAL.isolate_largest_group_original(ElementMat)
        It=list(self.VoidCheck).count(0)
        if rs_place==1 and action not in self.BC and SingleCheck[1]==True:
            done=False
            Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),self.Lx,self.Ly,self.EX,self.EY,self.Loaded_Node,self.Loaded_Node2,self.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)
            self.Max_SE_Ep=np.max(Run_Results[1])
            if abs(self.Max_SE_Tot/self.Max_SE_Ep)>=1 or abs(It/(self.EX*self.EY))>=1:
                reward=-1
                done=True
            else:
                reward = self.RS[(int((self.Max_SE_Tot/self.Max_SE_Ep)*1000))-1,int((It/(self.EX*self.EY))*1000)-1]
                reward2=self.RV[int(1-(np.reshape(self.Stress_state,(self.EX*self.EY,1))[action])*1000)-1]
                reward=statistics.mean([reward,reward2])
            self.Stress_state=Run_Results[3]
            self.Stress_state=np.reshape(self.Stress_state,(self.EX,self.EY))
            self.state=np.zeros((self.EX,self.EY,3))
            self.state[:,:,0]=self.Stress_state
            self.state[:,:,1]=self.BC_state
            self.state[:,:,2]=self.LC_state
        else:
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-10)"""
            Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),self.Lx,self.Ly,self.EX,self.EY,self.Loaded_Node,self.Loaded_Node2,self.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)
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
                self.VoidCheck[action]=1
            
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
        RenderMat[self.BC1_Element]=2
        RenderMat[self.BC2_Element]=2
        RenderMat[self.Loaded_Element]=4
        RenderMat=np.reshape(RenderMat,(self.EX,self.EY))
        print(np.flip(RenderMat,0))
        print('')
        return 
        
    def reset(self):

        self.VoidCheck=np.ones((1,self.EX*self.EY))
        self.VoidCheck=list(self.VoidCheck[0])
        self.VoidCheck=np.array(self.VoidCheck)
        self.Stress_state =FEA_SOLVER_GENERAL.FEASolve(list(np.ones((1,self.EX*self.EY))[0]),self.Lx,self.Ly,self.EX,self.EY,self.Loaded_Node,self.Loaded_Node2,self.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)[3]
        #self.Stress_state=list(np.array(self.Stress_state)
        self.Stress_state=np.reshape(self.Stress_state,(self.EX,self.EY))
        self.state=np.zeros((self.EX,self.EY,3))
        self.state[:,:,0]=self.Stress_state
        self.state[:,:,1]=self.BC_state
        self.state[:,:,2]=self.LC_state

        return self.state
    def reset_conditions(self):
        self.Max_SE_Tot=0
        self.VoidCheck=np.ones((1,self.EX*self.EY))
        self.VoidCheck=list(self.VoidCheck[0])
        self.VoidCheck=np.array(self.VoidCheck)
        while self.Max_SE_Tot<=0 or self.Max_SE_Tot>5000:
            self.BC1_Element=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            self.BC2_Element=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            while self.BC1_Element==self.BC2_Element:
                self.BC2_Element=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            self.BC1,self.BC2=BC_Nodes(self.BC1_Element,self.Lx,self.Ly,self.EX,self.EY)
            self.BC3,self.BC4=BC_Nodes(self.BC2_Element,self.Lx,self.Ly,self.EX,self.EY)
            self.BC_set=[self.BC1_Element,self.BC2_Element]
            #self.Loaded_Node=int(random.choice([i for i in Go_List]))
            self.Loaded_Element=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            while self.Loaded_Element in self.BC_set:
                self.Loaded_Element=int(random.choice([i for i in Element_Lists(self.EX,self.EY)[1]]))
            
            self.BC_set=np.append(self.BC_set,self.Loaded_Element)
            self.LC_state=list(np.zeros((1,self.EX*self.EY))[0])
            self.LC_state[self.Loaded_Element]=1
            self.LC_state=np.reshape(self.LC_state,(self.EX,self.EY))
            self.Load_Type=random.choice([0,1])
            self.Loaded_Node=LC_Nodes(self.Loaded_Element,self.Load_Type,self.Lx,self.Ly,self.EX,self.EY,Node_Location=False,)[0]
            self.Loaded_Node2=LC_Nodes(self.Loaded_Element,self.Load_Type,self.Lx,self.Ly,self.EX,self.EY,Node_Location=False)[1]
            if self.Load_Type==0: #Load will be applied vertically
                self.Loaded_Node+=((self.EX+1)*(self.EY+1))
                self.Loaded_Node2+=((self.EX+1)*(self.EY+1))
            self.Load_Direction=random.choice([-1,1]) #1 for Compressive Load, -1 for tensile load
            self.BC=[self.BC1_Element,self.BC2_Element,self.Loaded_Element]
            self.BC_state=list(np.zeros((1,self.EX*self.EY))[0])
            self.BC_state[self.BC1_Element]=1
            self.BC_state[self.BC2_Element]=1
            self.BC_state=np.reshape(self.BC_state,(self.EX,self.EY))
            self.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(self.VoidCheck,self.Lx,self.Ly,self.EX,self.EY,self.Loaded_Node,self.Loaded_Node2,self.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)[1]))
            self.Row_BC1=math.floor(self.BC1_Element/self.EX)
            self.Col_BC1=int(round(math.modf(self.BC1_Element/self.EX)[0]*self.EX,0))
            self.Row_BC2=math.floor(self.BC2_Element/self.EX)
            self.Col_BC2=int(round(math.modf(self.BC2_Element/self.EX)[0]*self.EX,0))
            self.Row_LC=math.floor(self.Loaded_Element/self.EX)
            self.Col_LC=int(round(math.modf(self.Loaded_Element/self.EX)[0]*self.EX,0))
            self.BC2_BC1=abs(self.Row_BC2-self.Row_BC1)+abs(self.Col_BC2-self.Col_BC1)
            self.BC2_LC=abs(self.Row_BC2-self.Row_LC)+abs(self.Col_BC2-self.Col_LC)
            self.BC1_LC=abs(self.Row_BC1-self.Row_LC)+abs(self.Col_BC1-self.Col_LC)
            self.Len_Mat=[self.BC2_BC1,self.BC2_LC,self.BC1_LC]
            self.Len_Mat.remove(max(self.Len_Mat))
            self.Min_Length=sum(self.Len_Mat)+1
    def primer_cond(self,EX,EY):
         self.BC=[self.BC1_Element,self.BC2_Element,self.Loaded_Element]
         self.BC_state=list(np.zeros((1,EX*EY))[0])
         self.BC_state[self.BC1_Element]=1
         self.BC_state[self.BC2_Element]=1
         self.BC_state=np.reshape(self.BC_state,(EX,EY))
         self.LC_state=list(np.zeros((1,EX*EY))[0])
         self.LC_state[self.Loaded_Element]=1
         self.LC_state=np.reshape(self.LC_state,(EX,EY))
         self.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(list(np.ones((1,self.EX*self.EY)))[0],self.Lx,self.Ly,self.EX,self.EY,self.Loaded_Node,self.Loaded_Node2,self.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)[1]))
def Prog_Refine_Act(agent_primer,env,env_primer,load_checkpoint,Testing,Lx,Ly,Small_EX,Small_EY,Big_EX,Big_EY):
    '''This function will deliver the optimal topology of the smaller sized environment.
    This final topology will then be transformed into the equivalent topology at the 
    larger selected size. This larger topology will then be based back to the main function
    and the topology removal process will continue.'''
    Stable=False
    while not Stable:
        env_primer.BC1,env_primer.BC2,env_primer.BC3,env_primer.BC4,env_primer.BC1_Element,env_primer.BC2_Element,env_primer.Loaded_Element,env_primer.Loaded_Node,env_primer.Loaded_Node2,env_primer.Min_Length,env_primer.Load_Direction=Condition_Transform(Lx,Ly,Small_EX,Small_EY,Big_EX,Big_EY,env.BC1,env.BC2,env.BC1_Element,env.BC2_Element,env.Loaded_Element,env.Load_Type,env.Load_Direction)
        LN_Hold=env_primer.Loaded_Node
        LN2_Hold=env_primer.Loaded_Node2
        if env.Load_Type==0:
            env_primer.Loaded_Node+=((Small_EX+1)*(Small_EY+1))
            env_primer.Loaded_Node2+=((Small_EX+1)*(Small_EY+1))
        env_primer.primer_cond(Small_EX,Small_EY)
        if env_primer.BC1==LN_Hold or env_primer.BC1==LN2_Hold or env_primer.BC2==LN_Hold or env_primer.BC2==LN2_Hold or env_primer.BC3==LN_Hold or env_primer.BC3==LN2_Hold or env_primer.BC4==LN_Hold or env_primer.BC4==LN2_Hold:
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
        action = agent_primer.choose_action(observation_primer,load_checkpoint,Testing=True)
        observation_primer_, reward, primer_done, It = env_primer.step(action,observation_primer,Last_Reward)
        observation_primer = observation_primer_
        Last_Reward=reward
        if primer_done and load_checkpoint and env_primer.Min_Length/((Small_EX*Small_EY)-list(env_primer.VoidCheck).count(0))<=.3:
            primer_done=False
            print(np.flip(np.reshape(range(0,(Small_EX)*(Small_EY)),(Small_EX,Small_EY)),0))
            action=int(input('It appears I am stuck, please suggest an element to remove: '))
            observation_primer_, reward, primer_done, It = env_primer.step(action,observation_primer,Small_EX,Small_EY,Last_Reward)
            observation_primer = observation_primer_
        if load_checkpoint:
            env_primer.render()
    env.VoidCheck=Mesh_Transform(Small_EX,Small_EY,Big_EX,Big_EY,env_primer.VoidCheck)
    Last_Reward=0
    if Testing:
        env_primer.render(Small_EX,Small_EY)
    if load_checkpoint:
        env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,Lx,Ly,Big_EX,Big_EY,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]))
        
def User_Inputs(env,Lx,Ly,Elements_X,Elements_Y):
    '''When testing a trained agent, the user will be prompted to select
    a single element to act as the loaded element, and two elements to act as the boundary 
    condition elements. Depending on where the elements are located, the nodes
    corresponding to these elements will be selected'''
    
    print(np.flip(np.reshape(range(0,(Elements_X)*(Elements_Y)),(Elements_X,Elements_Y)),0))
    env.BC1_Element=int(input('Please select an element to apply Boundary condition #1: '))
    if env.BC1_Element>(Elements_X)*(Elements_Y) or env.BC1_Element<0 or env.BC1_Element!=int(env.BC1_Element):
        print('Code Terminated By User...')
        sys.exit()
    env.BC2_Element=int(input('Please select a node to apply Boundary condition #2: '))
    if env.BC2_Element>(Elements_X)*(Elements_Y) or env.BC2_Element<0 or env.BC2_Element!=int(env.BC2_Element):
        print('Code Terminated By User...')
        sys.exit()
    print(np.flip(np.reshape(range(0,(Elements_X*Elements_Y)),(Elements_X,Elements_Y)),0))
    env.Loaded_Element=int(input('Please select an element to apply the load to: '))
    if env.Loaded_Element>(Elements_X)*(Elements_Y) or env.Loaded_Element<0 or env.Loaded_Element!=int(env.Loaded_Element):
        print('Code Terminated By User...')
        sys.exit()
    env.Load_Type=int(input('Input 0 for a Vertical Load or Input 1 for a Horizontal load: '))
    env.Load_Direction=int(input('Input -1 for a tensile load or Input 1 for a compressive load: '))
    env.Loaded_Node,env.Loaded_Node2=LC_Nodes(env.Loaded_Element,env.Load_Type,env.Lx,env.Ly,env.EX,env.EY,Node_Location=True)
    if env.Load_Type==0:
        env.Loaded_Node+=(Elements_X+1)*(Elements_Y+1)
        env.Loaded_Node2+=(Elements_X+1)*(Elements_Y+1)
    env.BC1,env.BC2=BC_Nodes(env.BC1_Element,env.Lx,env.Ly,env.EX,env.EY)
    env.BC3,env.BC4=BC_Nodes(env.BC2_Element,env.Lx,env.Ly,env.EX,env.EY)
    env.LC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
    env.LC_state[env.Loaded_Element]=1
    env.LC_state=np.reshape(env.LC_state,(Elements_X,Elements_Y))
    env.BC=[env.BC1_Element,env.BC2_Element,env.Loaded_Element]
    env.BC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
    env.BC_state[env.BC1_Element]=1
    env.BC_state[env.BC2_Element]=1
    env.BC_state=np.reshape(env.BC_state,(Elements_X,Elements_Y))
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]))
    env.Row_BC1=math.floor(env.BC1_Element/Elements_X)
    env.Col_BC1=int(round(math.modf(env.BC1_Element/Elements_X)[0]*Elements_X,0))
    env.Row_BC2=math.floor(env.BC2_Element/Elements_X)
    env.Col_BC2=int(round(math.modf(env.BC2_Element/Elements_X)[0]*Elements_X,0))
    env.Row_LC=math.floor(env.Loaded_Element/Elements_X)
    env.Col_LC=int(round(math.modf(env.Loaded_Element/Elements_X)[0]*Elements_X,0))
    env.BC2_BC1=abs(env.Row_BC2-env.Row_BC1)+abs(env.Col_BC2-env.Col_BC1)
    env.BC2_LC=abs(env.Row_BC2-env.Row_LC)+abs(env.Col_BC2-env.Col_LC)
    env.BC1_LC=abs(env.Row_BC1-env.Row_LC)+abs(env.Col_BC1-env.Col_LC)
    env.Len_Mat=[env.BC2_BC1,env.BC2_LC,env.BC1_LC]
    env.Len_Mat.remove(max(env.Len_Mat))
    env.Min_Length=sum(env.Len_Mat)+1
    return 
def Testing_Inputs(env,Lx,Ly,Elements_X,Elements_Y):
    '''Every 200 episodes, the boundary and loading conditions
    should be set as those of a cantilever beam to monitor the progress
    of the agents learning'''
    env.BC1=0
    env.BC2=env.BC1
    env.BC3=Elements_X*(Elements_Y+1)
    env.BC4=env.BC3
    env.Loaded_Node=Elements_X+(Elements_X+1)*(Elements_Y+1)
    env.Loaded_Node2=Elements_X-1+(Elements_X+1)*(Elements_Y+1)
    env.Loaded_Element=np.where(FEA_SOLVER_GENERAL.rectangularmesh(Lx,Ly,Elements_X,Elements_Y)[1]==env.Loaded_Node-((Elements_X+1)*(Elements_Y+1)))[0][0]
    env.BC1_Element=0
    env.BC2_Element=(Elements_X)*(Elements_Y-1)
    env.LC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
    env.LC_state[env.Loaded_Element]=1
    env.LC_state=np.reshape(env.LC_state,(Elements_X,Elements_Y))
    env.Load_Type=0
    env.Load_Direction=-1 #1 for Compressive Load, -1 for tensile load
    env.BC=[env.BC1_Element,env.BC2_Element,env.Loaded_Element]
    env.BC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
    env.BC_state[env.BC1_Element]=1
    env.BC_state[env.BC2_Element]=1
    env.BC_state=np.reshape(env.BC_state,(Elements_X,Elements_Y))
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]))

def Testing_Info(env,Lx,Ly,Elements_X,Elements_Y,score,Fixed,RN):
    '''Function that outputs the results of a testing trial. The results include
    the score based on the reward function, the final strain energy, and if needed
    the number of arbitrary blocks removed by the shaving algorithm'''
    if env.Load_Type==0:
        Load_Type='Vertical'
    else:
        Load_Type='Horizontal'
    if env.Load_Direction==-1:
        Load_Dir='Tensile'
    else:
        Load_Dir='Compressive'
    print('----------------')
    if not Fixed:
        print('The final topology with BCs located at elements '+str(env.BC1_Element)+' and '+str(env.BC2_Element)+' with a '+Load_Type+' '+Load_Dir+' load applied to element '+str(env.Loaded_Element)+': \n')
        print('BC1 Nodes: '+str(env.BC1)+' '+str(env.BC2))
        print('BC2 Nodes: '+str(env.BC3)+' '+str(env.BC4))
        print('LC Nodes: '+str(env.Loaded_Node)+' '+str(env.Loaded_Node2))
    else:
        print('The fixed topology with trivial elements removed: \n')
    env.render()
    if not Fixed:
        print('Episode Score: '+str(round(score,2)))
        print('Strain Energy: '+str(round(env.Max_SE_Ep,1)))
        print('----------------')
    else:
        print('Strain Energy for Trimmed Topology: '+str(round(np.max(FEA_SOLVER_GENERAL.FEASolve(list(env.VoidCheck),Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]),1)))
        print('Number of Extra Elements Removed: '+str(int(RN)))
        print('----------------')
        
def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G   

def Reward_Surface():
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
    m = np.linalg.lstsq(G, z)[0]
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
    mm = np.linalg.lstsq(GG, Z_Data)[0]

    GoGG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    Reward_Ind = np.reshape(np.dot(GoGG, mm), xx.shape)[:,-1]
    return zz,Reward_Ind