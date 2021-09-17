# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:34:50 2021

@author: nbrow
"""
import numpy as np
import scipy 
import math
from Node_Element_Extraction import BC_Nodes,LC_Nodes
def action_flip(action,Elements_X,Elements_Y):
    
    '''Given an element that is being loaded, produce the 
    element horizontally, vertically, and diagonally mirrored to it'''
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

def obs_flip(observation,Elements_X,Elements_Y):
    '''Given an observation, produce the observations 
    that are horizontally, vertically, and diagonally mirrored'''
    
    
    observation_v=np.zeros((Elements_X,Elements_Y,3))
    observation_h=np.zeros((Elements_X,Elements_Y,3))
    observation_vh=np.zeros((Elements_X,Elements_Y,3))
    for Flip in range(0,3):
        observation_v[:,:,Flip]=np.flip(observation[:,:,Flip],axis=0)
        observation_h[:,:,Flip]=np.flip(observation[:,:,Flip],axis=1)
    for Flip in range(0,3):
        observation_vh[:,:,Flip]=np.flip(observation_v[:,:,Flip],axis=1)
        
    return observation_v,observation_h,observation_vh

def Mesh_Triming(env,Elements_X,Elements_Y):
    '''This function can be used to eleminate elements that are only 
    singularly connected to the rest of the matrix and do not provide 
    substantial support to the rest of the structure. It can be thought
    of as a shaving algorithm to help catch single elements the RL agent mises 
    at the end'''
    
    Final=False
    Count_1=list(env.VoidCheck).count(0)
    while not Final:
        VC_Hold=np.zeros((Elements_X+2,Elements_Y+2))
        VC_Hold[1:Elements_X+1,1:Elements_Y+1]=np.reshape(env.VoidCheck,(Elements_X,Elements_Y))
        c = scipy.signal.convolve2d(VC_Hold, np.array([[0,1,0],[1,0,1],[0,1,0]]), mode='valid')
        VV=VC_Hold[1:Elements_X+1,1:Elements_Y+1]
        VV_Loc=np.where(np.reshape(VV,(1,(Elements_X*Elements_Y)))==0)[1]
        c=np.reshape(c,(1,Elements_X*Elements_Y))[0]
        c[VV_Loc]=0
        c_Loc=np.where(c==1)[0]
        for i in range(0,len(env.BC)):
            c_Loc=np.delete(c_Loc,np.where(c_Loc==env.BC[i]))
        if len(c_Loc)>0:
            Final=False
        else:
            Final=True

        if len(c_Loc)>0:
            try:
                env.VoidCheck[c_Loc]=0 
            except TypeError:
                env.VoidCheck[c_Loc[0]]=0
    Count_2=list(env.VoidCheck).count(0)
    return Count_2-Count_1  

def Condition_Transform(Lx,Ly,Old_EX,Old_EY,New_EX,New_EY,BC_Elements,LC_Elements,Load_Type,Load_Direction):
    New_BC_Elements=[]
    New_BC_Nodes=[]
    New_LC_Elements=[]
    New_LC_Nodes=[]
    for BC in range(0,len(BC_Elements)):
        BC1_E=BC_Elements[BC]
        Row_BC1_E=math.floor(BC1_E/New_EY)
        Col_BC1_E=math.floor(round(math.modf(BC1_E/New_EX)[0],2)*New_EX)

        Old_X_Perc_BC1_E=Row_BC1_E/New_EY
        Old_Y_Perc_BC1_E=Col_BC1_E/New_EX

        New_Row_BC1_E=math.floor((Old_X_Perc_BC1_E*Old_EX)+0.001)
        New_Col_BC1_E=math.floor((Old_Y_Perc_BC1_E*Old_EY)+0.001)
        New_BC1_E=(New_Row_BC1_E*Old_EX)+New_Col_BC1_E
        New_BC1,New_BC2=BC_Nodes(New_BC1_E,Lx,Ly,Old_EX,Old_EY)
        New_BC_Elements=np.append(New_BC_Elements,New_BC1_E)
        New_BC_Nodes=np.append(New_BC_Nodes,New_BC1)
        New_BC_Nodes=np.append(New_BC_Nodes,New_BC2)
    
    for LC_ in range(0,len(LC_Elements)):
        LC=LC_Elements[LC_]
        Row_LC=math.floor(LC/New_EY)
        Col_LC=math.floor(round(math.modf(LC/New_EX)[0],2)*New_EX)
        Old_X_Perc=Row_LC/New_EY
        Old_Y_Perc=Col_LC/New_EX
        New_Row_LC=math.floor((Old_X_Perc*Old_EX)+0.001)
        New_Col_LC=math.floor((Old_Y_Perc*Old_EY)+0.001)
        New_LC_E=(New_Row_LC*Old_EX)+New_Col_LC
        New_LC1,New_LC2=LC_Nodes(New_LC_E,Load_Type[LC_],Load_Direction[LC_],Lx,Ly,Old_EX,Old_EY,LC_,Node_Location=False)
        New_LC_Elements=np.append(New_LC_Elements,New_LC_E)
        New_LC_Nodes=np.append(New_LC_Nodes,New_LC1)
        New_LC_Nodes=np.append(New_LC_Nodes,New_LC2)
    
    return New_BC_Nodes,New_BC_Elements,New_LC_Elements,New_LC_Nodes,Load_Direction

def Mesh_Transform(Old_EX,Old_EY,New_EX,New_EY,Config):

    Config=np.reshape(Config,(Old_EX,Old_EY))
    Old_X_Perc=100/Old_EX
    Old_Y_Perc=100/Old_EY
    New_X_Perc=100/New_EX
    New_Y_Perc=100/New_EY
    
    New_Config=np.zeros((New_EY,New_EX))
    for i in range(Old_EX):
        for j in range(Old_EY):
            if Config[i,j]==1:
                Old_X_Min=j*Old_X_Perc
                Old_X_Max=(j+1)*Old_X_Perc
                Old_Y_Min=i*Old_Y_Perc
                Old_Y_Max=(i+1)*Old_Y_Perc
                New_X_Block_Max=math.ceil(Old_X_Max/New_X_Perc)
                if New_X_Block_Max!=0:
                    New_X_Block_Max-=1
                New_X_Block_Min=math.floor(Old_X_Min/New_X_Perc)
                #if New_X_Block_Min!=0:
                #    New_X_Block_Min-=1
                New_Y_Block_Max=math.ceil(Old_Y_Max/New_Y_Perc)
                if New_Y_Block_Max!=0:
                    New_Y_Block_Max-=1
                New_Y_Block_Min=math.floor(Old_Y_Min/New_Y_Perc)
                #if New_Y_Block_Min!=0:
                #    New_Y_Block_Min-=1
                New_Config[New_Y_Block_Min:New_Y_Block_Max+1,New_X_Block_Min:New_X_Block_Max+1:1]=1
    New_Config=np.reshape(New_Config,(1,New_EX*New_EY))
    
    return list(New_Config[0])
            
            
            
