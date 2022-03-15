# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:33:14 2022

@author: nbrow
"""


import numpy as np
from gym import Env
from gym.spaces import Discrete, Box
import random
from RL_Bezier import RL_Bezier_Design as RLBD
from Matrix_Transforms import isolate_largest_group_original
from RL_Resnet import create_res_net
import math
import matplotlib.pyplot as plt 

def Corner_Check(state):
    E_X=np.shape(state)[1]
    E_Y=np.shape(state)[0]
    Clear=False 
    if state[0,0]==1 and state[0,E_X-1]==1 and state[E_Y-1,0]==1 and state[E_Y-1,E_X-1]==1:
        Clear=True
    else:
        Clear=False
    return Clear

class UC_Env(Env):
    def __init__(self):
        self.action_space=Box(np.array([0,0,0,0,0,0,0]),np.array([1,1,1,1,1,1,1]))
        self.max_steps=7
        self.E_Y=20
        self.E_X=60
        self.Coef_Count=4
        self.Coef_Predictor=create_res_net()
        self.Coef_Predictor.load_weights('UC_Surrogate_Model_Weights_3-11-22\cp.ckpt')
        self.SS_Avg=np.load('Constant_Values/SS_Avg.npy')
        self.SS_Stdev=np.load('Constant_Values/SS_Stdev.npy')
        self.p_Stdev=[np.load('Constant_Values/pl1_Stdev.npy'),np.load('Constant_Values/pl2_Stdev.npy'),
                      np.load('Constant_Values/pl3_Stdev.npy'),np.load('Constant_Values/pl4_Stdev.npy')]
        self.p_Avg=[np.load('Constant_Values/pl1_Avg.npy'),np.load('Constant_Values/pl2_Avg.npy'),
                      np.load('Constant_Values/pl3_Avg.npy'),np.load('Constant_Values/pl4_Avg.npy')]
    def step(self,action):
        self.state_UC=RLBD(action,self.state_UC)

        self.step_count+=1
        SingleCheck=isolate_largest_group_original(self.state_UC)
        CornerCheck=Corner_Check(self.state_UC)
        if SingleCheck[1]==False and self.step_count>2 or CornerCheck==False  and self.step_count>2:
            Reward=-2
            Done=False
            Legal=False
            self.Coef_Current=[0,0,0,0]
        elif SingleCheck[1]==False and self.step_count<=2 or CornerCheck==False and self.step_count<=2:
            Reward=-2
            Done=False
            Legal=False
            self.Coef_Current=np.array([0,0,0,0])
        else:
            Legal=True
            self.state_UC=np.reshape(self.state_UC,(20,60))
            self.Coef_Current=self.Coef_Predictor.predict(np.reshape(self.state_UC,(1,20,60,1)))[0]
            self.Coef_Current[0]=math.atanh(self.Coef_Current[0])*(2*self.p_Stdev[0])+self.p_Avg[0]
            self.Coef_Current[1]=math.atanh(self.Coef_Current[1])*(2*self.p_Stdev[1])+self.p_Avg[1]
            self.Coef_Current[2]=math.atanh(self.Coef_Current[2])*(2*self.p_Stdev[2])+self.p_Avg[2]
            self.Coef_Current[3]=math.atanh(self.Coef_Current[3])*(2*self.p_Stdev[3])+self.p_Avg[3]
            self.Current_Fit=np.array([(Strain**3*self.Coef_Current[0])+(Strain**2*self.Coef_Current[1])+(Strain*self.Coef_Current[2])+self.Coef_Current[3] for Strain in self.Strain])
            self.Current_Fit[1:]=10**(self.Current_Fit[1:]*(2*self.SS_Stdev)+self.SS_Avg)
            self.Current_Fit[0]=0
            self.CC_Error=abs(np.mean([((i-j)/i) for i,j in zip(self.Current_Fit[1:len(self.Current_Fit)-1],self.Target_Fit[1:len(self.Target_Fit)-1])]))

            Reward=max(1/self.CC_Error,-1)
            if self.CC_Error<0.1:
                Done=True
            else:
                Done=False
        if self.step_count>self.max_steps:
            Done=True 
        return self.state_UC,self.Coef_Current, Reward, Done, Legal
            
    def render(self,Legal):
        self.state_UC=np.reshape(self.state_UC,(20,60))
        self.EX=np.shape(self.state_UC)[1]
        self.EY=np.shape(self.state_UC)[0]
        self.Element_Plot=np.zeros((self.EY*2,self.EX*2))
        self.Element_Plot[0:self.EY,0:self.EX]=self.state_UC
        self.Element_Plot[0:self.EY,self.EX:2*self.EX]=np.flip(self.state_UC,axis=1)
        self.Element_Plot[self.EY:self.EY*2,0:self.EX]=np.flip(self.state_UC,axis=0)
        self.Element_Plot[self.EY:self.EY*2,self.EX:2*self.EX]=np.flip(np.flip(self.state_UC,axis=0),axis=1)
        
        fig,ax= plt.subplots()
        ax.imshow(self.Element_Plot,cmap='Blues',origin='lower')
        ax.set_title('Current Proposed Unit-Cell Design')
        fig2,ax2=plt.subplots()
        if Legal:
            ax2.plot(self.Strain,self.Current_Fit,'-',label='Current Response')
            ax2.plot(self.Strain,self.Target_Fit,'--',label='Desired Response')
            ax2.legend()
        ax2.set_xlabel('Normalized Strain')
        ax2.set_ylabel('Stress')
        ax2.set_title('Stress-Strain Curve Comparison')
        
        
    def reset(self):
        self.step_count=0
        self.state_UC=np.zeros((20,60))
        self.state_Coef=np.zeros((1,4))
        #self.Coef_Target=[0.1391985 , -0.19501862, -0.13712262,  0.19724466]#[int(item) for item in input("Enter your 4 desired coefficient values with a space between each value. \n(Ex:  0.3 0.5 0.1 .99) \nEnter Coef: ").split()]
        self.Coef_Target=[0.30322606,-0.15345431, -0.06615746,  0.40950999]
   
        self.Coef_Target[0]=math.atanh(self.Coef_Target[0])*(2*self.p_Stdev[0])+self.p_Avg[0]
        self.Coef_Target[1]=math.atanh(self.Coef_Target[1])*(2*self.p_Stdev[1])+self.p_Avg[1]
        self.Coef_Target[2]=math.atanh(self.Coef_Target[2])*(2*self.p_Stdev[2])+self.p_Avg[2]
        self.Coef_Target[3]=math.atanh(self.Coef_Target[3])*(2*self.p_Stdev[3])+self.p_Avg[3]
        self.Final_Strain=1#float(input('Enter the final strain value between [0,0.2]:   '))
        self.Strain=np.linspace(0,self.Final_Strain,10)
        self.Current_Fit=[0]
        self.Target_Fit=np.array([(Strain**3*self.Coef_Target[0])+(Strain**2*self.Coef_Target[1])+(Strain*self.Coef_Target[2])+self.Coef_Target[3] for Strain in self.Strain])
        self.Target_Fit[1:]=10**(self.Target_Fit[1:]*(2*self.SS_Stdev)+self.SS_Avg)
        return self.state_UC, self.state_Coef
  
