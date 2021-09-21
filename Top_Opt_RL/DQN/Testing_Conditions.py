# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 12:39:07 2021

@author: nbrow
"""
import numpy as np
import math
import sys
from Top_Opt_RL.DQN. Node_Element_Extraction import BC_Nodes, LC_Nodes
import Top_Opt_RL.DQN. FEA_SOLVER_GENERAL as FEA_SOLVER_GENERAL
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