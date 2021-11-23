# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 16:39:01 2021

@author: nbrow
"""
import numpy as np
import matplotlib.pyplot as plt
import math 
import random
from FEA_SOLVER_GENERAL import *
import copy
from Abaqus_INP_writer import INP_writer
from skimage.measure import label
def label_( x): return label(x, connectivity=1)
def get_zero_label(d, labels):
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            if d[i,j]==0:
                return labels[i,j]

def isolate_largest_group_original(x):
    """Isolates the largest group. 
    this original version has problems with periodicity. The v2 version fixes this. """
    x_original = np.copy(x)    
    labels= label_(x)
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
E_X=20
E_Y=20
LX=1
LY=1
E_X*=LX
E_Y*=LY
Tesselate=False
Mirror=True 
if Tesselate:
    UX=3
    UY=3
else:
    UX=1
    UY=1
Random=True
Clear=False
FEA_Solve=False
Type='Corner'   #'Edge' for Top and Bottom Edges    'Corner' for the 4 Diagnoal corners   
               # Spec for Specific BCs and LCs Elements 


N_X=E_X+1
N_Y=E_Y+1
Element_Block=np.zeros(((E_X)*(E_Y),1))
Node_Num=N_X*N_Y
if Type=='Corner':
    Element_Block[0]=1
    Element_Block[E_X-1]=1
    Element_Block[E_X*(E_Y-1)]=1
    Element_Block[(E_X*E_Y)-1]=1
    

elif Type=='Edge':
    Element_Block[0:E_X]=1
    Element_Block[(E_X)*(E_Y-1):((E_X)*(E_Y))]=1
    
elif Type=='Spec':
    BC_Array=[[12,0],
              [12,23],
              [0,12],
              [23,12]]
              
    BCS=[]
    for BC_Count in range(0,len(BC_Array)):
        BCS=np.append(BCS,int((BC_Array[BC_Count][1]*E_X)+BC_Array[BC_Count][0]))
    for BC_Count in range(0,len(BCS)):
        Element_Block[int(BCS[BC_Count])]=1


lm=1
while lm<=2 or not Clear:
    if Random:
        if Type=='Edge':
            P_0=[random.randint(0,(E_X*LX)-1),0]
            P_3=[random.randint(0,(E_X*LX)-1),E_Y*LY]
        elif Type=='Corner':
            P_0=[random.choice([0,(E_X*LX)-1]),0]
            P_3=[random.choice([0,(E_X*LX)-1]),(E_Y*LY)-1]
        elif Type=='Spec':
            P_0=BC_Array[random.randint(0,len(BCS)-1)]
            P_3=BC_Array[random.randint(0,len(BCS)-1)]

        P_1=[random.randint(0,(E_X-1)),random.randint(0,(E_Y)-1)]
        P_2=[random.randint(0,(E_X-1)),random.randint(0,(E_Y)-1)]
    

        Thickness=random.choice([1,2])
    else:
        if lm==1:
            P_0=[1,10]
            P_1=[11,20]
            P_2=[23,10]
            Thickness=1

    
    P_0_aug=[P_0[0]/N_X,P_0[1]/N_Y]
    P_1_aug=[P_1[0]/N_X,P_1[1]/N_Y]
    P_2_aug=[P_2[0]/N_X,P_2[1]/N_Y]
    P_3_aug=[P_3[0]/N_X,P_3[1]/N_Y]
    t=np.arange(0, 1.01, 0.01).tolist()
    B_X=np.zeros((len(t),1))
    B_Y=np.zeros((len(t),1))
    
    for i in range(0,len(t)):
        B_X[i]=(((1-t[i])**3)*P_0_aug[0])+(3*((1-t[i])**2)*t[i]*P_1_aug[0])+(3*(1-t[i])*t[i]**2*P_2_aug[0])+(t[i]**3*P_3_aug[0])
        B_Y[i]=(((1-t[i])**3)*P_0_aug[1])+(3*((1-t[i])**2)*t[i]*P_1_aug[1])+(3*(1-t[i])*t[i]**2*P_2_aug[1])+(t[i]**3*P_3_aug[1])
    
    
    X_Seg=np.arange(0,LX,LX/(E_X+1))
    Y_Seg=np.arange(0,LY,LY/(E_Y+1))
    Segment_Mat=np.zeros((E_X,E_Y))
    Seg_Nodes=np.zeros((E_X*E_Y,4))
    d=0
    for j in range(0,E_Y):
        for i in range(0,E_X):
            Seg_Nodes[d,0]=X_Seg[i]
            Seg_Nodes[d,1]=X_Seg[i+1]
            Seg_Nodes[d,2]=X_Seg[j]
            Seg_Nodes[d,3]=X_Seg[j+1]
            d+=1
    
    X_Lower=[]
    X_Upper=[]
    Y_Lower=[]
    Y_Upper=[]
    Eliminate=[]
    Mat_Hold=copy.deepcopy(Element_Block)
    for k in range(0,len(B_X)-1):
        X_Test=(Seg_Nodes[:,0]<=B_X[k]) & (Seg_Nodes[:,1]>B_X[k])
        Y_Test=(Seg_Nodes[:,2]<=B_Y[k]) & (Seg_Nodes[:,3]>B_Y[k])
        Remove=[i for i, (x, y) in enumerate(zip(X_Test, Y_Test)) if x == y == True]
        
        Element_Block[Remove]=0
        if len(Remove)!=0:
            Loc=math.modf(Remove[0]/(N_X-1))
            Upper_Bound=int(min(int((N_Y-1)-(Loc[1]+0.001)),Thickness))
            Lower_Bound=int(min(int((Loc[1]+0.001)),Thickness))
            Left_Bound=int(min(int((N_X-1)*Loc[0]),Thickness))
            Right_Bound=int(min(int((N_X-1)-((N_X-1)*Loc[0])),Thickness))
        
            Element_Block[Remove[0]-Left_Bound:Remove[0]+Right_Bound]=1
            Element_Block[range(Remove[0]-(Lower_Bound*(N_X-1)),Remove[0]+(Upper_Bound*(N_X-1)),N_X-1)]=1
            Element_Block[Remove[0]-Lower_Bound*(N_X-1)-Left_Bound:Remove[0]-Lower_Bound*(N_X-1)+Right_Bound]=1
            Element_Block[Remove[0]+Upper_Bound*(N_X-1)-Left_Bound:Remove[0]+Upper_Bound*(N_X-1)+Right_Bound]=1
            if Thickness==2:
                Element_Block[Remove[0]-max(Lower_Bound-1,0)*(N_X-1)-Left_Bound:Remove[0]-max(Lower_Bound-1,0)*(N_X-1)+Right_Bound]=1
                Element_Block[Remove[0]+max(Upper_Bound-1,0)*(N_X-1)-Left_Bound:Remove[0]+max(Upper_Bound-1,0)*(N_X-1)+Right_Bound]=1
    SingleCheck=isolate_largest_group_original(np.reshape(Element_Block,(E_Y,E_X)))
    
    if SingleCheck[1]==True:
        Clear=True
    lm+=1
    
if Mirror and not Tesselate:
    Element_Plot=np.zeros((E_Y*2,E_X*2))
    Element_Plot[0:E_Y,0:E_X]=np.reshape(Element_Block,(E_Y,E_X))
    Element_Plot[0:E_Y,E_X:2*E_X]=np.flip(np.reshape(Element_Block,(E_Y,E_X)),axis=1)
    Element_Plot[E_Y:E_Y*2,0:E_X]=np.flip(np.reshape(Element_Block,(E_Y,E_X)),axis=0)
    Element_Plot[E_Y:E_Y*2,E_X:2*E_X]=np.flip(np.flip(np.reshape(Element_Block,(E_Y,E_X)),axis=0),axis=1)
    fig,ax= plt.subplots()
    ax.imshow(Element_Plot,cmap='Blues',origin='lower')
elif Mirror and Tesselate:
    Element_Mir=np.zeros((E_Y*2,E_X*2))
    Element_Mir[0:E_Y,0:E_X]=np.reshape(Element_Block,(E_Y,E_X))
    Element_Mir[0:E_Y,E_X:2*E_X]=np.flip(np.reshape(Element_Block,(E_Y,E_X)),axis=1)
    Element_Mir[E_Y:E_Y*2,0:E_X]=np.flip(np.reshape(Element_Block,(E_Y,E_X)),axis=0)
    Element_Mir[E_Y:E_Y*2,E_X:2*E_X]=np.flip(np.flip(np.reshape(Element_Block,(E_Y,E_X)),axis=0),axis=1)
    if not Overlap:
        Element_Plot=np.ones(((E_Y)*2*UY,(E_X)*2*UX))
    else:
        Element_Plot=np.ones(((E_Y)*2*UY,(E_X)*2*UX))
    for y_move in range(0,UY):
        for x_move in range(0,UX):
            Element_Plot[(E_Y*2)*y_move:(E_Y*2)*(y_move+1),(E_X*2)*(x_move):(E_X*2)*(x_move+1)]=Element_Mir
    fig,ax= plt.subplots()
    ax.imshow(Element_Plot,cmap='Blues',origin='lower')
    
elif Tesselate and not Mirror:
    Element_Plot=np.ones(((N_Y-1)*UY,(N_X-1)*UX))
    for y_move in range(0,UY):
        for x_move in range(0,UX):
            Element_Plot[(N_Y-1)*y_move:(N_Y-1)*(y_move+1),(N_X-1)*(x_move):(N_X-1)*(x_move+1)]=np.reshape(Element_Block,(N_Y-1,N_X-1))

    fig,ax= plt.subplots()
    ax.imshow(Element_Plot,cmap='Blues',origin='lower')


elif not Tesselate and not Mirror:
    Element_Plot=np.reshape(Element_Block,(E_Y,E_X))
    fig,ax= plt.subplots()
    ax.imshow(Element_Plot,cmap='Blues',origin='lower')
    
[nodeCoor,elementNodes]=rectangularmesh(E_X*2*UX,E_Y*2*UY,E_X*2*UX,E_X*2*UY)
Mat_Loc=(np.reshape(Element_Plot,((E_X*2*E_Y*2*UX*UY),1))==1)
Mat_Keep=[i for i, x in enumerate(Mat_Loc) if x == True]
Elements=np.zeros((len(Mat_Keep),5))
Elements[:,0:4]=elementNodes[Mat_Keep,:]
Elements[:,4]=[int(x+1) for x in Mat_Keep]
Nodes_Loc=list(set(np.reshape(Elements[:,0:4],(len(Elements)*4)).astype('int')))
Nodes=np.zeros((len(Nodes_Loc),3))
Nodes[:,0:2]=nodeCoor[Nodes_Loc,:]
Nodes[:,2]=[int(x+1) for x in Nodes_Loc]
Elements[:,0:4]+=1
INP_writer(Elements,Nodes,E_X,E_Y,UX,UY)


    

    
    
    
    
    
    
