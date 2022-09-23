# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:59:26 2022

@author: nbrow
"""
import numpy as np
import random 
import matplotlib.pyplot as plt 
import math
from Matrix_Transforms import isolate_largest_group_original
import copy 
def RL_Bezier_Design(action,state_UC):
    if type(action)!=np.ndarray:
        action=action.numpy()

    E_Y=np.shape(state_UC)[0]
    E_X=np.shape(state_UC)[1]

    

    if action[0]<=0.5:
        P_0=[0,0]
    else:
        P_0=[0,E_X-1]

    if action[1]<=0.5:
        P_3=[E_Y-1,0]
    else:
        P_3=[E_Y-1,E_X-1]
     


    P_1=[math.floor(action[2]*(E_Y-1)),math.floor(action[3]*(E_X-1))]
    P_2=[math.floor(action[4]*(E_Y-1)),math.floor(action[5]*(E_X-1))]

    if action[6]<0.5:
        Thickness=1
        Choice=0
    elif action[6]>=0.5:
        Choice=1
        Thickness=1


    #Build=round(action[7])
    P_0_aug=[P_0[0]/E_Y,P_0[1]/E_X]
    P_1_aug=[P_1[0]/E_Y,P_1[1]/E_X]
    P_2_aug=[P_2[0]/E_Y,P_2[1]/E_X]
    P_3_aug=[P_3[0]/E_Y,P_3[1]/E_X]

    t=np.arange(0, 1.01, 0.01).tolist()
    B_X=np.zeros((len(t),1))
    B_Y=np.zeros((len(t),1))
    
    for i in range(0,len(t)):
        B_Y[i]=(((1-t[i])**3)*P_0_aug[0])+(3*((1-t[i])**2)*t[i]*P_1_aug[0])+(3*(1-t[i])*t[i]**2*P_2_aug[0])+(t[i]**3*P_3_aug[0])
        B_X[i]=(((1-t[i])**3)*P_0_aug[1])+(3*((1-t[i])**2)*t[i]*P_1_aug[1])+(3*(1-t[i])*t[i]**2*P_2_aug[1])+(t[i]**3*P_3_aug[1])

    X_Seg=np.arange(0,1,1/(E_X+1))
    Y_Seg=np.arange(0,1,1/(E_Y+1))
    Seg_Nodes_X=np.zeros((E_X,2))
    Seg_Nodes_Y=np.zeros((E_Y,2))
    for i in range(0,E_X):
        Seg_Nodes_X[i,0]=X_Seg[i]
        Seg_Nodes_X[i,1]=X_Seg[i+1]

        
    for j in range(0,E_Y):
        Seg_Nodes_Y[j,0]=Y_Seg[j]
        Seg_Nodes_Y[j,1]=Y_Seg[j+1]
    Action_UC=np.zeros((20,60))
    for k in range(0,len(B_X)-1):
        X_Test=(Seg_Nodes_X[:,0]<=B_X[k]) & (Seg_Nodes_X[:,1]>B_X[k])
        Y_Test=(Seg_Nodes_Y[:,0]<=B_Y[k]) & (Seg_Nodes_Y[:,1]>B_Y[k])
        Remove_X=np.where(X_Test==True)

        if type(Remove_X) is not 'numpy.int64':
            Remove_X=Remove_X[0][0]
        Remove_Y=np.where(Y_Test==True)
        if type(Remove_Y) == tuple:
            Remove_Y=Remove_Y[0][0]

        Loc_X=math.modf(Remove_X/(E_X))
        Loc_Y=math.modf(Remove_Y/(E_Y))
        Upper_Bound=int(min(int((E_Y)-(Loc_Y[1]+0.0001)),Thickness))
        Lower_Bound=int(min(int(Loc_Y[1]+0.0001),Thickness))
        Left_Bound=int(min(int((E_X)*Loc_X[0]),Thickness))
        Right_Bound=int(min(int((E_X)-((E_X)*Loc_X[0])),Thickness))
        UC_Hold=copy.deepcopy(state_UC)
        if Choice==0:
            state_UC[Remove_Y-Lower_Bound:Remove_Y+Upper_Bound,Remove_X-Left_Bound:Remove_X+Right_Bound]=1
            Action_UC[Remove_Y-Lower_Bound:Remove_Y+Upper_Bound,Remove_X-Left_Bound:Remove_X+Right_Bound]=1

        else:
            state_UC[Remove_Y-Lower_Bound:Remove_Y+Upper_Bound+1,Remove_X-Left_Bound:Remove_X+Right_Bound+1]=1
            Action_UC[Remove_Y-Lower_Bound:Remove_Y+Upper_Bound+1,Remove_X-Left_Bound:Remove_X+Right_Bound+1]=1
        SingleCheck=isolate_largest_group_original(Action_UC)
        if SingleCheck[1]==False:
            state_UC[Remove_Y-1:Remove_Y+1,Remove_X-1:Remove_X+1]=1
    return state_UC