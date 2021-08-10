# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""


''' Nathan Brown 
Policy Gradient Training of Topology Optimization through Reinforcement learning'''
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers,models 
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K
import sys
import pandas as pd
import matplotlib.pyplot as plt
import time
import statistics 
from Condition_Transformer import Condition_Transform
from Mesh_Transformer import Mesh_Transform
import random
import math
from gym import Env
from gym.spaces import Discrete
import FEA_SOLVER_GENERAL
import itertools
import copy 
import scipy
from keract import get_activations, display_activations 

def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G        
def LC_Nodes(Load_Element,Load_Direction):
    #Given the loaded element and corresponding loading direction, produce the nodes that should be loaded for the FEA
    Load_Nodes=ElementNodes[Load_Element,:]
    if Load_Element in Bottom_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]
    if Load_Element in Top_List:
        Loaded_Node=Load_Nodes[2]
        Loaded_Node2=Load_Nodes[3]
    if Load_Element in Right_List:
        Loaded_Node=Load_Nodes[1]
        Loaded_Node2=Load_Nodes[2]
    if Load_Element in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[3]
    if Load_Element in Top_List and Load_Element in Right_List:
        if Load_Direction==1:
            Loaded_Node=Load_Nodes[1]
            Loaded_Node2=Load_Nodes[2]
        else:
            Loaded_Node=Load_Nodes[2]
            Loaded_Node2=Load_Nodes[3]
    if Load_Element in Top_List and Load_Element in Left_List:
        if Load_Direction==1:
            Loaded_Node=Load_Nodes[0]
            Loaded_Node2=Load_Nodes[3]
        else:
            Loaded_Node=Load_Nodes[2]
            Loaded_Node2=Load_Nodes[3]
    if Load_Element in Bottom_List and Load_Element in Right_List:
        if Load_Direction==1:
            Loaded_Node=Load_Nodes[1]
            Loaded_Node2=Load_Nodes[2]
        else:
            Loaded_Node=Load_Nodes[0]
            Loaded_Node2=Load_Nodes[1]
    if Load_Element in Bottom_List and Load_Element in Left_List:
        if Load_Direction==1:
            Loaded_Node=Load_Nodes[0]
            Loaded_Node2=Load_Nodes[3]
        else:
            Loaded_Node=Load_Nodes[0]
            Loaded_Node2=Load_Nodes[1]
    if Load_Element not in Bottom_List and Load_Element not in Top_List and Load_Element not in Right_List and Load_Element not in Left_List:
        Dir=random.randrange(0,4)
        if Dir==0:
            Loaded_Node=Load_Nodes[0]
            Loaded_Node2=Load_Nodes[1]
        if Dir==1:
            Loaded_Node=Load_Nodes[1]
            Loaded_Node2=Load_Nodes[2]
        if Dir==2:
            Loaded_Node=Load_Nodes[2]
            Loaded_Node2=Load_Nodes[3]
        if Dir==3:
            Loaded_Node=Load_Nodes[3]
            Loaded_Node2=Load_Nodes[0]
    Loaded_Node=int(Loaded_Node)
    Loaded_Node2=int(Loaded_Node2)   
    return Loaded_Node, Loaded_Node2
def BC_Nodes(Load_Element):
    
    # Given the Boundary Condition Element,produce the corresponding nodes depending on where it's located
    Load_Nodes=ElementNodes[Load_Element,:]
    if Load_Element in Bottom_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]
    if Load_Element in Top_List:
        Loaded_Node=Load_Nodes[2]
        Loaded_Node2=Load_Nodes[3]
    if Load_Element in Right_List:
        Loaded_Node=Load_Nodes[1]
        Loaded_Node2=Load_Nodes[2]
    if Load_Element in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[3]
    if Load_Element in Top_List and Load_Element in Right_List:
        Loaded_Node=Load_Nodes[2]
        Loaded_Node2=Load_Nodes[2]
    if Load_Element in Top_List and Load_Element in Left_List:
        Loaded_Node=Load_Nodes[3]
        Loaded_Node2=Load_Nodes[3]
    if Load_Element in Bottom_List and Load_Element in Right_List:
        Loaded_Node=Load_Nodes[1]
        Loaded_Node2=Load_Nodes[1]

    if Load_Element in Bottom_List and Load_Element in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[0]
    if Load_Element not in Bottom_List and Load_Element not in Top_List and Load_Element not in Right_List and Load_Element not in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]

    Loaded_Node=int(Loaded_Node)
    Loaded_Node2=int(Loaded_Node2)   
    return Loaded_Node, Loaded_Node2
def action_flip(action):
    'Given an element that is being loaded, produce the element horizontally and vertically symmetrical'
    Action_Mat=np.zeros((1,Elements_X*Elements_Y))
    Action_Mat[0][action]=1
    Action_Mat=np.reshape(Action_Mat,(Elements_X,Elements_Y))
    AM_v=np.reshape(np.flip(Action_Mat,axis=0),(1,Elements_X*Elements_Y))
    AM_h=np.reshape(np.flip(Action_Mat,axis=1),(1,Elements_X*Elements_Y))
    AM_vh=np.reshape(np.flip(np.reshape(AM_v,(Elements_X,Elements_Y)),axis=1),(1,Elements_X*Elements_Y))
    action_v=np.where(AM_v[0]==1)[0][0]
    action_h=np.where(AM_h[0]==1)[0][0]
    action_vh=np.where(AM_vh[0]==1)[0][0]
    return action_v,action_h,action_vh  
def obs_flip(observation):
    'Given an observation, produce the observations that are vertically and horizontally mirrored'
    observation_v=np.zeros((Elements_X,Elements_Y,3))
    observation_h=np.zeros((Elements_X,Elements_Y,3))
    observation_vh=np.zeros((Elements_X,Elements_Y,3))
    for Flip in range(0,3):
        observation_v[:,:,Flip]=np.flip(observation[:,:,Flip],axis=0)
        observation_h[:,:,Flip]=np.flip(observation[:,:,Flip],axis=1)
    for Flip in range(0,3):
        observation_vh[:,:,Flip]=np.flip(observation_v[:,:,Flip],axis=1)
        
    return observation_v,observation_h,observation_vh
def Element_Lists(Elements_X,Elements_Y):
    Go_List=[]
    Elem_List=[]
    Go_List=np.append(Go_List,range(0,Elements_X+1))
    Elem_List=np.append(Elem_List,range(0,Elements_X))

    for num in range(0,Elements_Y-1):
        Go_List=np.append(Go_List,(Elements_X+1)*(num+1))
        Go_List=np.append(Go_List,(Elements_X+1)*(num+2)-1)
    Go_List=np.append(Go_List,range((Elements_X*(Elements_Y+1)),(Elements_X*(Elements_Y+2)+1)))
    for num in range(0,Elements_Y-2):
        Elem_List=np.append(Elem_List,(Elements_X*(num+1)))
        Elem_List=np.append(Elem_List,(Elements_X*(num+2)-1))
    Elem_List=np.append(Elem_List,range(Elements_X*(Elements_Y-1),(Elements_X*(Elements_Y))))
    Bottom_List=np.arange(0, Elements_X,1).tolist()
    Top_List=np.arange(Elements_X*(Elements_Y-1),Elements_X*Elements_Y, 1).tolist()
    Left_List=np.arange(0, Elements_X*(Elements_Y),Elements_X).tolist()
    Right_List=np.arange(Elements_X-1,Elements_X*Elements_Y+1,Elements_X).tolist()

    return Go_List, Elem_List, Bottom_List, Top_List,Left_List,Right_List
def Reward_Surface():
    x=np.array([1,0,0,1,.5,0,.5])
    y=np.array([0,0,1,1,.5,.5,0])
    z=np.array([])
    a=5
    b=5
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
def User_Inputs():
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
    env.Loaded_Node=LC_Nodes(env.Loaded_Element,env.Load_Type)[0]
    env.Loaded_Node2=LC_Nodes(env.Loaded_Element,env.Load_Type)[1]
    if env.Load_Type==0:
        env.Loaded_Node+=NodeSize
        env.Loaded_Node2+=NodeSize
    env.BC1,env.BC2=BC_Nodes(env.BC1_Element)
    env.BC3,env.BC4=BC_Nodes(env.BC2_Element)
    env.LC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
    env.LC_state[env.Loaded_Element]=1
    env.LC_state=np.reshape(env.LC_state,(Elements_X,Elements_Y))
    env.BC=[env.BC1_Element,env.BC2_Element,env.Loaded_Element]
    env.BC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
    env.BC_state[env.BC1_Element]=1
    env.BC_state[env.BC2_Element]=1
    env.BC_state=np.reshape(env.BC_state,(Elements_X,Elements_Y))
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]))
    return 
def Testing_Inputs():
    env.BC1=0
    env.BC2=env.BC1
    env.BC3=Elements_X*(Elements_Y+1)
    env.BC4=env.BC3
    env.Loaded_Node=Elements_X+(NodeSize)
    env.Loaded_Node2=Elements_X-1+NodeSize
    env.Loaded_Element=np.where(ElementNodes==env.Loaded_Node-NodeSize)[0][0]
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
    env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]))
def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episodes')
    plt.ylabel(' Average Reward')
    plt.savefig(figure_file)
def Data_History(Total_Loss,score):
    Loss_history.append(Total_Loss)
    avg_Loss=np.mean(Loss_history[-50:])
    score_history.append(score)
    avg_score = np.mean(score_history[-50:])
    Succ_Steps=list(env.VoidCheck).count(0)
    succ_history.append(Succ_Steps)

    avg_succ = np.mean(succ_history[-50:])
    Percent_Succ=Succ_Steps/ElementSize
    per_history.append(Percent_Succ)
    avg_percent=np.mean(per_history[-50:])
    return Succ_Steps,Percent_Succ,avg_succ,avg_score,avg_Loss,avg_percent
def Testing_Info(score,EX,EY,Fixed,RN):
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
    else:
        print('The fixed topology with trivial elements removed: \n')
    env.render(EX,EY)
    if not Fixed:
        print('Episode Score: '+str(round(score,2)))
        print('Strain Energy: '+str(round(env.Max_SE_Ep,1)))
        print('----------------')
    else:
        print('Strain Energy for Trimmed Topology: '+str(round(np.max(FEA_SOLVER_GENERAL.FEASolve(list(env.VoidCheck),Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]),1)))
        print('Number of Extra Elements Removed: '+str(int(RN)))
        print('----------------')
def Mesh_Triming(ElementsX,ElementsY):
    Final=False
    Count_1=list(env.VoidCheck).count(0)
    while not Final:
        VC_Hold=np.zeros((ElementsX+2,ElementsY+2))
        VC_Hold[1:ElementsX+1,1:ElementsY+1]=np.reshape(env.VoidCheck,(ElementsX,ElementsY))
        c = scipy.signal.convolve2d(VC_Hold, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='valid')
        VV=VC_Hold[1:ElementsX+1,1:ElementsY+1]
        VV_Loc=np.where(np.reshape(VV,(1,(ElementsX*ElementsY)))==0)[1]
        c=np.reshape(c,(1,ElementsX*ElementsY))[0]
        c[VV_Loc]=0
        c_Loc=np.where(c==1)[0]
        for i in range(0,len(env.BC)):
            c_Loc=np.delete(c_Loc,np.where(c_Loc==env.BC[i]))
        if len(c_Loc)>0:
            Final=False
        else:
            Final=True
        print(c_Loc)
        if len(c_Loc)>0:
            try:
                env.VoidCheck[c_Loc]=0 
            except TypeError:
                env.VoidCheck[c_Loc[0]]=0
    Count_2=list(env.VoidCheck).count(0)
    return Count_2-Count_1
    
class Between(Constraint):
   def __init__(self, min_value=1, max_value=1):
       self.min_value = min_value
       self.max_value = max_value

   def __call__(self, w):        
       return K.clip(w, self.min_value, self.max_value)

   def get_config(self):
       return {'min_value': self.min_value,
               'max_value': self.max_value}           
'General Input'
Elements_X=11
Elements_Y=11
P_X=5
P_Y=5
Num_hidden=128
Ep_decay=5e-4
Mem_Size=10000
Lx=1
Ly=1

filename_save = 'DDQN_TopOpt_Generalized_CNN_4L_Gen_'+str(Elements_X)+'by'+str(Elements_Y)
#filename_save = 'DDQN_TopOpt_Generalized_CNN_4L_10by10'

#filename_load = 'DDQN_TopOpt_Generalized_CNN_4L_Gen_'+str(P_X)+'by'+str(P_Y)
filename_load = 'DDQN_TopOpt_Generalized_CNN_4L_Gen_5by5'

'House Keeping'
VoidCheck=np.ones((1,Elements_X*Elements_Y))
VoidCheck=list(VoidCheck[0])
ElementSize=Elements_X*Elements_Y
NodeSize=(Elements_X+1)*(Elements_Y+1)
ElementNodes=FEA_SOLVER_GENERAL.rectangularmesh(Lx,Ly,Elements_X,Elements_Y)[1]

tic=time.perf_counter()
Go_List,Elem_List,Bottom_List,Top_List,Left_List,Right_List=Element_Lists(Elements_X,Elements_Y)
zz,Reward_Ind=Reward_Surface()

class TopOpt_Gen(Env):
    def __init__(self):
        #Actons we can take... remove any of the blocks
        self.action_space=Discrete(ElementSize)
        self.eta=2
    def step(self,action,observation,EX,EY,Last_Reward):
        #Apply Action
            
        # evaluate it on grid
      
        rs_place=self.VoidCheck[action]
        self.VoidCheck[action]=0
        ElementMat=np.reshape(self.VoidCheck,(EX,EY))
        SingleCheck=FEA_SOLVER_GENERAL.isolate_largest_group_original(ElementMat)
        It=list(self.VoidCheck).count(0)
        if rs_place==1 and action not in self.BC and SingleCheck[1]==True:
            done=False
            Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),Lx,Ly,EX,EY,self.Loaded_Node,self.Loaded_Node2,env.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)
            self.Max_SE_Ep=np.max(Run_Results[1])
            if abs(self.Max_SE_Tot/self.Max_SE_Ep)>=1 or abs(It/(EX*EY))>=1:
                reward=-1
                done=True
            else:
                reward = zz[(int((self.Max_SE_Tot/self.Max_SE_Ep)*1000))-1,int((It/(EX*EY))*1000)-1]
                reward2=Reward_Ind[int(1-(np.reshape(self.Stress_state,(EX*EY,1))[action])*1000)-1]
                reward=statistics.mean([reward,reward2])
            self.Stress_state=Run_Results[3]
            self.Stress_state=np.reshape(self.Stress_state,(EX,EY))
            self.state=np.zeros((EX,EY,3))
            self.state[:,:,0]=self.Stress_state
            self.state[:,:,1]=self.BC_state
            self.state[:,:,2]=self.LC_state
        else:
            """If the removed block has already been removed, leads to a non-singular
            body or one of the Boundary condition blocks, the agent should be severely punished (-10)"""
            Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),Lx,Ly,EX,EY,self.Loaded_Node,self.Loaded_Node2,env.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)
            self.Max_SE_Ep=np.max(Run_Results[1])
            self.Stress_state=Run_Results[3]
            self.Stress_state=np.reshape(self.Stress_state,(EX,EY))
            self.state=np.zeros((EX,EY,3))
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
    
    def render(self,EX,EY, mode='human'):
        RenderMat=copy.deepcopy(self.VoidCheck)
        RenderMat[self.BC1_Element]=2
        RenderMat[self.BC2_Element]=2
        RenderMat[self.Loaded_Element]=4
        RenderMat=np.reshape(RenderMat,(EX,EY))
        print(np.flip(RenderMat,0))
        print('')
        return 
        
    def reset(self,EX,EY):

        self.VoidCheck=np.ones((1,EX*EY))
        self.VoidCheck=list(self.VoidCheck[0])
        self.VoidCheck=np.array(self.VoidCheck)
        self.Stress_state =FEA_SOLVER_GENERAL.FEASolve(list(np.ones((1,EX*EY))[0]),Lx,Ly,EX,EY,self.Loaded_Node,self.Loaded_Node2,env.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)[3]
        #self.Stress_state=list(np.array(self.Stress_state)
        self.Stress_state=np.reshape(self.Stress_state,(EX,EY))
        self.state=np.zeros((EX,EY,3))
        self.state[:,:,0]=self.Stress_state
        self.state[:,:,1]=self.BC_state
        self.state[:,:,2]=self.LC_state

        return self.state
    def reset_conditions(self):
        self.Max_SE_Tot=0
        while self.Max_SE_Tot<=0 or self.Max_SE_Tot>5000:
            self.BC1_Element=int(random.choice([i for i in Elem_List]))
            self.BC2_Element=int(random.choice([i for i in Elem_List]))
            while self.BC1_Element==self.BC2_Element:
                self.BC2_Element=int(random.choice([i for i in Elem_List]))
            self.BC1,self.BC2=BC_Nodes(self.BC1_Element)
            self.BC3,self.BC4=BC_Nodes(self.BC2_Element)
            self.BC_set=[self.BC1_Element,self.BC2_Element]
            #self.Loaded_Node=int(random.choice([i for i in Go_List]))
            self.Loaded_Element=int(random.choice([i for i in Elem_List]))
            while self.Loaded_Element in self.BC_set:
                self.Loaded_Element=int(random.choice([i for i in Elem_List]))
            
            self.BC_set=np.append(self.BC_set,self.Loaded_Element)
            self.LC_state=list(np.zeros((1,Elements_X*Elements_Y))[0])
            self.LC_state[self.Loaded_Element]=1
            self.LC_state=np.reshape(self.LC_state,(Elements_X,Elements_Y))
            self.Load_Type=random.choice([0,1])
            self.Loaded_Node=LC_Nodes(self.Loaded_Element,self.Load_Type)[0]
            self.Loaded_Node2=LC_Nodes(self.Loaded_Element,self.Load_Type)[1]
            if self.Load_Type==0: #Load will be applied vertically
                self.Loaded_Node+=NodeSize
                self.Loaded_Node2+=NodeSize
            self.Load_Direction=random.choice([-1,1]) #1 for Compressive Load, -1 for tensile load
            self.BC=[self.BC1_Element,self.BC2_Element,self.Loaded_Element]
            self.BC_state=list(np.zeros((1,Elements_X*Elements_Y))[0])
            self.BC_state[self.BC1_Element]=1
            self.BC_state[self.BC2_Element]=1
            self.BC_state=np.reshape(self.BC_state,(Elements_X,Elements_Y))
            self.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,self.Loaded_Node,self.Loaded_Node2,self.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)[1]))
    def primer_cond(self,EX,EY):
         self.BC=[self.BC1_Element,self.BC2_Element,self.Loaded_Element]
         self.BC_state=list(np.zeros((1,EX*EY))[0])
         self.BC_state[self.BC1_Element]=1
         self.BC_state[self.BC2_Element]=1
         self.BC_state=np.reshape(self.BC_state,(EX,EY))
         self.LC_state=list(np.zeros((1,EX*EY))[0])
         self.LC_state[self.Loaded_Element]=1
         self.LC_state=np.reshape(self.LC_state,(EX,EY))
         self.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(list(np.ones((1,EX*EY)))[0],Lx,Ly,EX,EY,self.Loaded_Node,self.Loaded_Node2,env.Load_Direction,self.BC1,self.BC2,self.BC3,self.BC4,Stress=True)[1]))

env = TopOpt_Gen()
env_primer= TopOpt_Gen()

class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions,Retrain,Increase):
        super(DuelingDeepQNetwork, self).__init__()
        self.model = models.Sequential()
        #self.model.add(layers.Conv2D(64,(3,3),trainable=Retrain,padding='same',activation='relu'))
        #self.model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
        if Increase:
            self.model.add(layers.Conv2D(64,(3,3),padding='same',activation='relu'))
            self.model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))

        self.model.add(layers.Conv2D(16,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(8,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(4,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(1,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Flatten())

    def call(self, state):
        x = self.model(state)

        #V = self.model_V(x)
        #A = self.model_A(x)
        
        Q = x#V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))
        return Q

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
    def __init__(self,Retrain,Increase, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims,filename_save,filename_load,EX,EY, epsilon_dec=Ep_decay, eps_end=0.01, 
                 mem_size=Mem_Size, replace=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions=n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.EX=EX
        self.EY=EY
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size
        self.lr=lr
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(self.EX*self.EY,Retrain,Increase)
        self.q_next = DuelingDeepQNetwork(self.EX*self.EY,Retrain,Increase)
        self.checkpoint_file_save='NN_Weights/'+filename_save+'_NN_weights'
        self.checkpoint_file_load='NN_Weights/'+filename_load+'_NN_weights'
        self.q_eval.compile(optimizer=Adam(learning_rate=self.lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=self.lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation,load_checkpoint,Testing):
        self.action_space = [i for i in range(self.n_actions)]
        if np.random.random() < self.epsilon and load_checkpoint is False and Testing is False:
            Void=np.array(env.VoidCheck)
            BC=np.array(np.reshape(env.BC_state,(1,(self.EX*self.EY))))
            LC=np.array(np.reshape(env.LC_state,(1,(self.EX*self.EY))))
            Clear_List=np.where(Void==0)[0]
            BC_List=np.where(BC==1)[0]
            LC_List=np.where(LC==1)[0]
            self.action_space = [ele for ele in self.action_space if ele not in Clear_List]
            self.action_space = [ele for ele in self.action_space if ele not in BC_List]
            self.action_space = [ele for ele in self.action_space if ele not in LC_List]
            action = np.random.choice(self.action_space)
        else:
            state = observation
            state=state.reshape(-1,self.EX,self.EY,3)
            actions = self.q_eval.call(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            Loss=.5
            return Loss

        if self.learn_step_counter % self.replace == 0 and self.learn_step_counter>0:
            agent.q_next.set_weights(agent.q_eval.get_weights())  

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_eval(states)
        self.q_pred=q_pred
        q_next = self.q_next(states_)
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
        if self.learn_step_counter>5000:
            self.lr=2.5e-3
        if self.learn_step_counter>7500:
            self.lr=1e-3
        return Loss
        
    def save_models(self):
        print('... saving models ...')
        self.q_eval.save_weights(self.checkpoint_file_save)

    def load_models(self):
        print('... loading models ...')
        self.q_eval.load_weights(self.checkpoint_file_load)
'---------------------------------------'



       
load_checkpoint=True #Make load_checkpoint True to test out 'filename_load' weights
Retrain=False  #Make Retrain True to train on a larger state space 
TimeTrial=False



'------------------------------------------'
if not load_checkpoint:
    print('You are about to train new weights... Click enter to continue..')
    input('')

agent_primer= Agent(Retrain,Increase=False,lr=5e-3, gamma=0.1,filename_save=filename_save,filename_load=filename_load,EX=P_X,EY=P_Y, n_actions=env.action_space.n, epsilon=0,
                  batch_size=128, input_dims=[P_X,P_Y,3])
agent_primer.load_models()
if load_checkpoint:
    agent = Agent(Retrain,Increase=False,lr=5e-3, gamma=0.1,filename_save=filename_save,filename_load=filename_load,EX=Elements_X,EY=Elements_Y, n_actions=env.action_space.n, epsilon=1.0,
                  batch_size=128, input_dims=[Elements_X,Elements_Y,3])
else:
    agent = Agent(Retrain,Increase=False,lr=5e-3, gamma=0.1,filename_save=filename_save,filename_load=filename_load,EX=Elements_X,EY=Elements_Y, n_actions=env.action_space.n, epsilon=1.0,
                  batch_size=128, input_dims=[Elements_X,Elements_Y,3])
agent.load_models()
n_games = 100_000
figure_file = 'plots/' + filename_save +'_reward.png'    
best_score = env.reward_range[0]
best_test_score=0
total_score=0
score_history = []
step_history=[]
per_history=[]
succ_history=[]
Loss_history=[]
Ep_History=[]
if load_checkpoint or Retrain or TimeTrial:
    agent.load_models()

if not load_checkpoint:
    TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Avg Loss','SDEV','Epsilon','Time'])
env.reset_conditions()
for i in range(n_games):
    Testing = False
    Stable=False
    if load_checkpoint:
        'If the user wants to test the agent, the user will be prompted to input BC and LC elements'
        User_Inputs()
    done = False
    score = 0
    if TimeTrial:
        TT_Start=time.perf_counter()
    if i%10==0 and i>=100:
        Testing=True
        if i%200==0:
            'Every 200 episodes, a special BC/LC will be used for monitoring purposes'
            Testing_Inputs()
            print('--------Testing Run------')
    while not Stable:
        observation = env.reset(Elements_X,Elements_Y)
        env_primer.BC1,env_primer.BC2,env_primer.BC3,env_primer.BC4,env_primer.BC1_Element,env_primer.BC2_Element,env_primer.Loaded_Element,env_primer.Loaded_Node,env_primer.Loaded_Node2=Condition_Transform(P_X,P_Y,Elements_X,Elements_Y,env.BC1,env.BC2,env.BC1_Element,env.BC2_Element,env.Loaded_Element,env.Load_Type)
        LN_Hold=env_primer.Loaded_Node
        LN2_Hold=env_primer.Loaded_Node2
        if env.Load_Type==0:
            env_primer.Loaded_Node+=((P_X+1)*(P_Y+1))
            env_primer.Loaded_Node2+=((P_X+1)*(P_Y+1))
        env_primer.primer_cond(P_X,P_Y)
        if env_primer.BC1==LN_Hold or env_primer.BC1==LN2_Hold or env_primer.BC2==LN_Hold or env_primer.BC2==LN2_Hold or env_primer.BC3==LN_Hold or env_primer.BC3==LN2_Hold or env_primer.BC4==LN_Hold or env_primer.BC4==LN2_Hold:
            env.reset_conditions()
        else:
            Stable=True
    primer_done=False
    observation_primer=env_primer.reset(P_X,P_Y)
    Last_Reward=0
    while not primer_done:
        action = agent_primer.choose_action(observation_primer,load_checkpoint,Testing=True)
        observation_primer_, reward, primer_done, It = env_primer.step(action,observation_primer,P_X,P_Y,Last_Reward)
        observation_primer = observation_primer_
        Last_Reward=reward
        #if load_checkpoint and i>=1:
        #    activations = get_activations(agent_primer.q_eval.model, observation_primer.reshape(-1,P_X,P_Y,3))
        #    display_activations(activations, save=False)
        if load_checkpoint and not TimeTrial:
            env_primer.render(P_X,P_Y)
    env.VoidCheck=Mesh_Transform(P_X,P_Y,Elements_X,Elements_Y,env_primer.VoidCheck)
    Last_Reward=0
    if Testing:
        env_primer.render(P_X,P_Y)
    if load_checkpoint:
        env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[1]))
    observation[:,:,0]=np.reshape(FEA_SOLVER_GENERAL.FEASolve(env.VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Loaded_Node2,env.Load_Direction,env.BC1,env.BC2,env.BC3,env.BC4,Stress=True)[3],(Elements_X,Elements_Y))
    observation_v, observation_h,observation_vh=obs_flip(observation)
    Last_Reward=0
    while not done:
        if i%1000==0 and i>=1:# or load_checkpoint and i>=1:
            activations = get_activations(agent.q_eval.model, observation.reshape(-1,Elements_X,Elements_Y,3))
            display_activations(activations, save=False)
        action = agent.choose_action(observation,load_checkpoint,Testing)
        tic2=time.perf_counter()
        observation_, reward, done, It= env.step(action,observation,Elements_X,Elements_Y,Last_Reward)
        observation_v_,observation_h_,observation_vh_=obs_flip(observation_)
        action_v,action_h,action_vh=action_flip(action)
        agent.store_transition(observation,action,reward,observation_,done)
        agent.store_transition(observation_v,action_v,reward,observation_v_,done)
        agent.store_transition(observation_h,action_h,reward,observation_h_,done)
        agent.store_transition(observation_vh,action_vh,reward,observation_vh_,done)
        score += reward
        Last_Reward=reward
        if Testing is True:
            env.render(Elements_X,Elements_Y)
            print('Current Score: '+str(round(score,3)))
        observation = observation_
        observation_v=observation_v_
        observation_h=observation_h_
        observation_vh=observation_vh_
        if load_checkpoint and not TimeTrial:
            env.render(Elements_X,Elements_Y)

    if load_checkpoint:
        Testing_Info(score,Elements_X,Elements_Y,Fixed=False,RN=0)
        Removed_Num=Mesh_Triming(Elements_X,Elements_Y)   
        Testing_Info(score,Elements_X,Elements_Y,Fixed=True,RN=Removed_Num)
    if not load_checkpoint:
        Total_Loss=agent.learn()
    else:
        Total_Loss=1
    Succ_Steps,Percent_Succ,avg_succ,avg_score,avg_Loss,avg_percent=Data_History(Total_Loss,score)
    if Testing:
        print('Current Best Testing Score: '+str(round(best_test_score,3)))
    if not load_checkpoint:
        env.reset_conditions()
    if avg_score>=best_score and not load_checkpoint:
        '''If the average score of the previous runs is better than 
        the previous best average then the new model should be saved'''
        agent.save_models()
        best_score=avg_score
    if Testing is True and not load_checkpoint and score>best_test_score and i>1:
        best_test_score=score
        print('New Best Testing Reward')
        agent.save_models()
    toc=time.perf_counter()
    if TimeTrial:
        print('The optimization process took '+str(round(toc-TT_Start,3))+' seconds to produce the final topology: ')
        env.render(Elements_X,Elements_Y)
    if not load_checkpoint:
        TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
                'Percent Successful':Percent_Succ,'Avg Loss':avg_Loss,'Epsilon': agent.epsilon, 'Time':round((toc-tic),3)}, ignore_index=True)
        
        print('Episode ', i, '  Score %.2f' % score,'  Avg_score %.2f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percent %.0f' %(avg_percent*100),'     Avg Loss %.2f' %avg_Loss,'  Ep.  %.2f' %agent.epsilon,'  Time (s) %.0f' %(toc-tic))
    if i%100==0 and not load_checkpoint:
        TrialData.to_pickle('Trial_Data/'+filename_save +'_TrialData.pkl')
    if not load_checkpoint and i%100==0 and i>0 :
        x = range(0,i+1)
        plot_learning_curve(x, score_history, figure_file)
 
