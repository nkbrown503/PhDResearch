# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 09:36:44 2021

@author: nbrow
"""
"""This code will be used to train a Reinforcement Learning Q-table
that will be updated using the Bellman Optimality Equation. The goal of the 
reinforcement learning agent is to minimize the compliance of a given structure
The agent will be tasked with removing certain blocks in a 4by 4 material matrix
to achieve the same results as a topology optimization algorithm. The compliance will
be determined using the 2D_linear_FEA_voided.py code which has been developed
by Nathan Brown in January of 2020."""

"""The main components of the FEA solver have been adjusted to be iteratively run 
to efficienctly train the Q-learning agent. The non-iterative FEA solver
can be found at 2D_linear_FEA_voided.py or 2D_linear_FEA.py"""

import time
import numpy as np
import matplotlib.pyplot as plt 
import sys
#import cupy
import random as rand
import warnings
import skimage.measure as measure
import pandas as pd
import pickle
if not sys.warnoptions:
    warnings.simplefilter("ignore")
tic=time.perf_counter()



'Define the Function that creates the rectangular mesh'
def label( x): return measure.label(x, connectivity=1)

def get_zero_label(d, labels):
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i,j]==0:
                return labels[i,j]

def isolate_largest_group_original(x):
    """Isolates the largest group. 
    this original version has problems with periodicity. The v2 version fixes this. """
    x_original = np.copy(x)    
    labels= label(x)
    zerolabel=get_zero_label(x,labels)
    group_names = np.unique(labels)      
    group_names = [g for g in group_names if g!=zerolabel]   
    vols = np.array([(labels==name).sum() for name in group_names])
    largestgroup_name = group_names[np.argmax(vols)]    
    x = labels == largestgroup_name
    x = x.astype("int")
    design_ok = np.all(x==x_original)  
    return x, design_ok

def rectangularmesh(Lx,Ly,ElementsX,ElementsY):
    Nx=ElementsX
    Ny=ElementsY
    xx=np.linspace(0,Lx,ElementsX+1)
    yy=np.linspace(0,Ly,ElementsY+1)
    nodeCoor=np.zeros(((Nx+1)*(Ny+1),2))
    elementNodes=np.zeros((Nx*Ny,4))
    for j in range(0,len(yy)):
        for i in range(0,len(xx)):
            call=(Nx+1)*(j)+i
            nodeCoor[call,0]=xx[i]
            nodeCoor[call,1]=yy[j]     
    for j in range(0,Ny):
        for i in range(0,Nx):
            d=(Nx)*(j)+i;
            elementNodes[d,0]=(Nx+1)*(j)+i
            elementNodes[d,1]=(Nx+1)*(j)+i+1
            elementNodes[d,2]=(Nx+1)*(j+1)+i+1
            elementNodes[d,3]=(Nx+1)*(j+1)+i
    return nodeCoor, elementNodes

def shapeFunctionQ4(xi,eta):
    shape=(1/4)*np.matrix([[(1-xi)*(1-eta)],
                           [(1+xi)*(1-eta)],
                           [(1+xi)*(1+eta)],
                           [(1-xi)*(1+eta)]])
    naturalDeriv=(1/4)*np.matrix([[-(1-eta),-(1-xi)],
                                  [ (1-eta),-(1+xi)],
                                  [ (1+eta), (1+xi)],
                                  [-(1+eta), (1-xi)]])
    return(shape,naturalDeriv) 

#-----------------------------------------------
'Determine Jacobian Matrix and Inverse of Jacobian'
def Jacobian(nodeCoor,naturalDeriv):
    JacobianMatrix=np.transpose(nodeCoor)*naturalDeriv
    invJacobian=np.linalg.inv(JacobianMatrix)
    XYDerivative=naturalDeriv*invJacobian
    return(JacobianMatrix,invJacobian,XYDerivative)

'Calculation of the System Stiffness Matrix'
def formStiffness2D(GDof,TotElements,numberNodes,elementNodes,nodeCoor,materials,rho,thickness,VoidCheck):
    """Compute Stiffness Matrix and (mass matrix)
       for plane stress Q4 elements"""
    stiffness=np.zeros((GDof,GDof))

    mass=np.zeros((GDof,GDof))
    
    '2 by 2 quadrature'
    gaussLocations=np.array([[-.577350269189626, -.577350269189626],
                             [ .577350269189626, -.577350269189626],
                             [ .577350269189626,  .577350269189626],
                             [-.577350269189626,  .577350269189626]])
    gaussWeights=np.array([[1],[1],[1],[1]])

    for e in range(0,TotElements):
        indice=(elementNodes[e,:])
        indice=indice.astype(int)
        elementDof=[]
        elementDof=np.append(indice,indice+numberNodes)
        ndof=len(indice)
        'Shape Functions and Derivatives'
    
        'Cycle for Gauss Point'
        for q in range(0,len(gaussWeights)):
            GaussPoint=gaussLocations[q,:]
            xi=GaussPoint[0]
            eta=GaussPoint[1]
            #Determine Shape Functions and Derivatives 
            sFQ4=shapeFunctionQ4(xi,eta)
            naturalDeriv=sFQ4[1]
            
            #Determine Jacobian
            JR=Jacobian(nodeCoor[indice,:],naturalDeriv)
            JacobianMatrix=JR[0]
            
            #invJacobian=JR[1]
            XYderivative=JR[2]
              #------------------------------------------------
            'Create the B matrix'
            B=np.zeros((3,2*ndof))
            B[0,0:ndof]         =np.transpose(XYderivative[:,0])
            B[1,ndof:(2*ndof)]  =np.transpose(XYderivative[:,1])
            B[2,0:ndof]         =np.transpose(XYderivative[:,1])
            B[2,ndof:(2*ndof)]  =np.transpose(XYderivative[:,0])
        
              #-------------------------------------------------
            'Stiffness matrix'
            'Assign Unique Material Properties Depending on Void or Material'
            if VoidCheck[e]==1:
                E=materials[0,0]
                poisson=materials[0,1]
            else:
                E=materials[1,0]
                poisson=materials[1,1]
            C=np.matrix([[1,poisson,0], [poisson,1,0],[0,0,(1-poisson)/2]])*(E/(1-(poisson**2)))
            Mat1=np.asarray(np.transpose(B)*C*thickness*B*1*np.linalg.det(JacobianMatrix))
         

            stiffness[np.ix_(elementDof,elementDof)]=stiffness[np.ix_(elementDof,elementDof)]+Mat1
            
    return(stiffness,mass)
def solution(GDof,prescribedDof,stiffness,force):
    # Function to find solution in terms of global dispalcements
    activeDof=[]
    GDof_list=list(range(0,GDof))
    for i in GDof_list:
        if i not in prescribedDof[0]:
            activeDof.append(i)
    ActiveStiffness=np.zeros((len(activeDof),len(activeDof)))

    ActiveStiffness=stiffness[np.ix_(activeDof,activeDof)]
    ActiveForce=np.zeros((len(activeDof),1))
    for i in range(0,len(activeDof)):
        ActiveForce[i]=force[activeDof[i]]
    
    U=np.matmul(np.linalg.inv(ActiveStiffness),ActiveForce)
    
    displacements=np.zeros((GDof,1))
    displacements[activeDof]=U
    return displacements

def FEASolve(VoidCheck):
    'Input Linear Material Properties'
    materials=np.zeros((2,2))
    #Real Material Properties
    materials[0,0]=10e9 #Modulus of Elasticity
    materials[0,1]=0.3 #Poisson's Ratio
    
    #Void Material Representation 
    """Taken from: An FEM Analysis with Consideration of Random Void Defects for 
    Predicting the Mechanical Properties of 3D Braided Composities, Kun Xu and Xiaomei Qian"""
    materials[1,0]=1 #Modulus of Elasticity (1e-6 MPa)
    materials[1,1]=0.000001 #Poisson's Ratio
    
    'Input Desired Loading Condition'
    P=-5e5#Magnitude
    
    'Mesh Generation'
    Lx=1 #Length in x direction
    Ly=1 #Length in y direction
    ElementsX=4 #Number of Elements in X Direction
    ElementsY=4 #Number of Elements in Y Direction
    TotElements=ElementsX*ElementsY #Total Number of Elements 
    RectOutput=rectangularmesh(Lx,Ly,ElementsX,ElementsY)
    nodeCoor=(RectOutput[0])
    elementNodes=(RectOutput[1])
    xx=nodeCoor[:,0]
    yy=nodeCoor[:,1]
    numberNodes=len(xx)
    'Global Number of Degrees of Freedom'
    GDof=2*numberNodes

        #-------------------------------------------------------------------
    'Boundary Conditions'
    '''This portion can be adjusted in accordance with the necessary boundary
    conditions of the given problem'''
    
    'BC currently set up to limit displacement on Y-Axis with distributed force on X=Lx' 
    fixedNodeX=[i for i,x in enumerate(nodeCoor[:,0]) if x==0]  
    fixedNodeY=[i for i,x in enumerate(nodeCoor[:,0]) if x==0]
     
    prescribedDof=[fixedNodeX,fixedNodeY+(np.ones(len(fixedNodeY))*numberNodes)]
    prescribedDof=[prescribedDof[0]+list(prescribedDof[1])]
    'Input Force Vectors'
    #Currently Assuming Distributed load applied at xx=Lx (Tensile Testing)
    force=np.zeros((GDof,1))
    
    
    #--------------------------------------------------------
    #|                                                      |
    #|                                                      |
    #|    Apply for Point Load on Bottom Right Corner       |
    #|                        |                             |
    #|                        v                             |
    #--------------------------------------------------------
    force[(ElementsX+1)*(ElementsY+1)+ElementsX]=P 
    FS2D=formStiffness2D(GDof,TotElements,numberNodes,elementNodes,nodeCoor,materials,1,1,VoidCheck)                                                         
    stiffness=FS2D[0]
    displacements=solution(GDof,prescribedDof,stiffness,force)
    MaxDisplacements=displacements.min()
    StrainEnergy=np.matmul(np.transpose(displacements),stiffness)
    StrainEnergy=np.matmul(StrainEnergy,displacements)
    return(MaxDisplacements,StrainEnergy)

#Define The Size and Scope of Your Training
Elements_X=4
Elements_Y=4
ElementSize=Elements_X*Elements_Y
Vol_fraction=10/16
Remove_num=ElementSize-(ElementSize*Vol_fraction)

#---Loading State Table----------------
State_Table=pd.read_pickle("4by4_StateTable.pkl")
State_Table=State_Table.to_numpy()
LeftBC=list(range(0,Elements_X*(Elements_Y-1)*2,Elements_X*(Elements_Y-1)))
LeftBC.append(Elements_X-1) #For Bottom Right
VoidCheck=np.ones((1,ElementSize))
VoidCheck=list(VoidCheck[0])
FailCheck=0
Tot_Reward=0
i=0
# Build Q-Table

#Hyperparameters
gamma=0.7
alpha=0.1

Train_Cycles=150000
tic=time.perf_counter()



"""---------------CHECK BEFORE EACH RUN----------------------"""

Train=False #Make this true if you need to retrain the q_table
Retrain=False
Save=False

Test=True


"""---------------------------------------------------------"""


if Train==True:
    if Retrain==True:
        q_table=np.load('Trained_Q_150k_4by4.npy')
    else:
        q_table=np.zeros((len(State_Table),ElementSize))
        q_table[:,LeftBC]=-100
    for j in range(0,Train_Cycles):
        #Reset Key Parameters After Every Run
        epsilon=(Train_Cycles-j)/Train_Cycles
        VoidCheck=np.ones((1,ElementSize))
        VoidCheck=list(VoidCheck[0])
        if j % 500 ==0:
            print('Current Episode:'+str(j))
            toc=time.perf_counter()
            print('Current Time [sec]: '+str(round(toc-tic,3)))
        i=0
        EndRun=False
        while i<int(Remove_num):
            Current_State=(np.where(np.all(State_Table==VoidCheck,axis=1)))
            Current_State=Current_State[0][0]            
            if rand.uniform(0,1)<epsilon: #Random Exploration
                removespot=rand.randint(0,ElementSize-1)
                rs_place=VoidCheck[removespot]
                VoidCheck[removespot]=0
                ElementMat=np.reshape(VoidCheck,(Elements_X,Elements_Y))
                SingleCheck=isolate_largest_group_original(ElementMat)
                #Make sure the material being removed exists and its not the Top or Bottom Part of the Left BC
                if rs_place==1 and removespot not in LeftBC and SingleCheck[1]==True:
                    VoidCheck[removespot]=0
                    i=i+1
                    Reward=FEASolve(list(VoidCheck))[1]*(-1e-4)#Check to ensure A Single Piece
                else: 
                    q_table[Current_State,removespot]=new_value=-100
                    i=Remove_num+1
                    EndRun=True
            else:
                removespot=np.argmax(q_table[Current_State])
                rs_place=VoidCheck[removespot]
                VoidCheck[removespot]=0
                ElementMat=np.reshape(VoidCheck,(Elements_X,Elements_Y))
                SingleCheck=isolate_largest_group_original(ElementMat)
                #Make sure the material being removed exists and its not the Top or Bottom Part of the Left BC
                if rs_place==1 and removespot not in LeftBC and SingleCheck[1]==True:
                    VoidCheck[removespot]=0
                    i=i+1
                    Reward=FEASolve(list(VoidCheck))[1]*(-1e-4)#Check to ensure A Single Piece
                else: 
                    q_table[Current_State,removespot]=new_value=-100
                    i=Remove_num+1
                    EndRun=True
            if EndRun==False:   
                #Using the Bellman Optimality Equation to Solve for the New Q Value
                New_State=(np.where(np.all(State_Table==VoidCheck,axis=1)))
                New_State=New_State[0][0]
                old_value=q_table[Current_State,removespot]
                next_max=np.max(q_table[New_State])
                new_value=(1-alpha)*old_value+alpha*(Reward+gamma*next_max)
                q_table[Current_State,removespot]=new_value
                Current_State=New_State

if Save==True:
    np.save('Trained_Q_150k_StrainEnergy_IterativeEp_4by4.npy',q_table)


#Run a test to compare these results to the "99 Line Topology Optimization Paper by 0. Sigmund"
if Test==True:
    Trained_Q_table=np.load('Trained_Q_150k_StrainEnergy_IterativeEp_4by4.npy')
    VoidCheck=np.ones((1,ElementSize))
    VoidCheck=list(VoidCheck[0])
    for i in range(0,int(Remove_num)):
        Current_State=(np.where(np.all(State_Table==VoidCheck,axis=1)))
        Current_State=Current_State[0][0]
        print(Current_State)
        removespot=np.argmax(Trained_Q_table[Current_State])
        VoidCheck[removespot]=0
    ElementMat=np.reshape(VoidCheck,(Elements_X,Elements_Y))
    print(ElementMat)






