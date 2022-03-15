# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:50:46 2021

@author: nbrow
"""

import numpy as np
import math
from skimage.measure import label
import sys
import matplotlib.pyplot as plt 
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
def Mesh_Transform(Config, Mesh_Complexity):

    New_Config=np.zeros((Config.shape[0]*Mesh_Complexity,Config.shape[1]*Mesh_Complexity))
    for i in range(Config.shape[0]):
        for j in range(Config.shape[1]):
            if Config[i,j]==1:
                New_Config[i*Mesh_Complexity:i*Mesh_Complexity+(Mesh_Complexity),j*Mesh_Complexity:j*Mesh_Complexity+Mesh_Complexity]=1
    
    return New_Config

def Fillet_Array(Mesh_Complexity,Matrix):
    TL=np.zeros(((Mesh_Complexity*2)+1,(Mesh_Complexity*2)+1))
    TR=np.zeros(((Mesh_Complexity*2)+1,(Mesh_Complexity*2)+1))
    BL=np.zeros(((Mesh_Complexity*2)+1,(Mesh_Complexity*2)+1))
    BR=np.zeros(((Mesh_Complexity*2)+1,(Mesh_Complexity*2)+1))
    #Top Left Filter
    TR[Mesh_Complexity,Mesh_Complexity]=0.1
    TR[Mesh_Complexity,0:Mesh_Complexity]=0.00001 #Right Line 
    TR[0:(Mesh_Complexity),Mesh_Complexity]=0.001 #Bottom Line 
    TR[0:Mesh_Complexity,0:Mesh_Complexity]=1
    # Top Right Filter
    TL[Mesh_Complexity,Mesh_Complexity]=0.1
    TL[0:(Mesh_Complexity),Mesh_Complexity]=0.001 #Bottom Line 
    TL[Mesh_Complexity,Mesh_Complexity+1:(Mesh_Complexity*2)+1]=0.0001 #Left Line
    TL[0:Mesh_Complexity,Mesh_Complexity+1:(2*Mesh_Complexity)+1]=1
    # Bottom Right Filter
    BL[Mesh_Complexity,Mesh_Complexity]=0.1
    BL[Mesh_Complexity+1:(Mesh_Complexity*2)+1,Mesh_Complexity]=0.01 #Top Line
    BL[Mesh_Complexity,Mesh_Complexity+1:(Mesh_Complexity*2)+1]=0.0001 #Left Line
    BL[Mesh_Complexity+1:(Mesh_Complexity*2)+1,Mesh_Complexity+1:(Mesh_Complexity*2)+1]=1
    # Bottom Left Filter
    BR[Mesh_Complexity,Mesh_Complexity]=0.1
    BR[Mesh_Complexity,0:Mesh_Complexity]=0.00001 #Right Line 
    BR[Mesh_Complexity+1:(Mesh_Complexity*2)+1,Mesh_Complexity]=0.01 #Top Line
    BR[Mesh_Complexity+1:(Mesh_Complexity*2)+1,0:Mesh_Complexity]=1

    Matrix_Pad=np.zeros((Matrix.shape[0]+(Mesh_Complexity*2),Matrix.shape[1]+(Mesh_Complexity*2)))
    Matrix_Pad[Mesh_Complexity:Matrix.shape[0]+Mesh_Complexity,Mesh_Complexity:Matrix.shape[1]+Mesh_Complexity]=Matrix

    TLCorners=np.zeros((Matrix.shape[0],Matrix.shape[1]))
    TRCorners=np.zeros((Matrix.shape[0],Matrix.shape[1]))
    BLCorners=np.zeros((Matrix.shape[0],Matrix.shape[1]))
    BRCorners=np.zeros((Matrix.shape[0],Matrix.shape[1]))

    for i in range(0,Matrix.shape[0]):
        for j in range(0,Matrix.shape[1]):
            TLCorners[i,j]=round(sum(sum(np.multiply(Matrix_Pad[i:i+(Mesh_Complexity*2)+1,j:j+(Mesh_Complexity*2)+1],TL))),6)
            TRCorners[i,j]=round(sum(sum(np.multiply(Matrix_Pad[i:i+(Mesh_Complexity*2)+1,j:j+(Mesh_Complexity*2)+1],TR))),6)
            BLCorners[i,j]=round(sum(sum(np.multiply(Matrix_Pad[i:i+(Mesh_Complexity*2)+1,j:j+(Mesh_Complexity*2)+1],BL))),6)
            BRCorners[i,j]=round(sum(sum(np.multiply(Matrix_Pad[i:i+(Mesh_Complexity*2)+1,j:j+(Mesh_Complexity*2)+1],BR))),6)

    TL_Loc=np.where(TLCorners==round(0.1+(Mesh_Complexity*0.001)+(Mesh_Complexity*0.0001),6))

    TR_Loc=np.where(TRCorners==round(0.1+(Mesh_Complexity*0.001)+(Mesh_Complexity*0.00001),6))
    
    BL_Loc=np.where(BLCorners==round(0.1+(Mesh_Complexity*0.01)+(Mesh_Complexity*0.0001),6))

    BR_Loc=np.where(BRCorners==round(0.1+(Mesh_Complexity*0.01)+(Mesh_Complexity*0.00001),6))

    if len(TL_Loc[0])==len(BL_Loc[0]):
        for i in range(0,len(TL_Loc[0])):
            Matrix[TL_Loc[0][i]-1,TL_Loc[1][i]:TL_Loc[1][i]+Mesh_Complexity]=1
            Matrix[TL_Loc[0][i]-(Mesh_Complexity-1):TL_Loc[0][i],TL_Loc[1][i]+1]=1
            Matrix[TL_Loc[0][i]-(Mesh_Complexity-2),TL_Loc[1][i]+(Mesh_Complexity-2)]=1
            Matrix[TR_Loc[0][i]-1,TR_Loc[1][i]-(Mesh_Complexity-1):TR_Loc[1][i]]=1
            Matrix[TR_Loc[0][i]-(Mesh_Complexity-1):TR_Loc[0][i],TR_Loc[1][i]-1]=1
            Matrix[TR_Loc[0][i]-(Mesh_Complexity-2),TR_Loc[1][i]-(Mesh_Complexity-2)]=1
            Matrix[BL_Loc[0][i]+1,BL_Loc[1][i]:BL_Loc[1][i]+Mesh_Complexity]=1
            Matrix[BL_Loc[0][i]:BL_Loc[0][i]+(Mesh_Complexity),BL_Loc[1][i]+1]=1
            Matrix[BL_Loc[0][i]+(Mesh_Complexity-2),BL_Loc[1][i]+(Mesh_Complexity-2)]=1
            Matrix[BR_Loc[0][i]+1,BR_Loc[1][i]-(Mesh_Complexity-1):BR_Loc[1][i]]=1
            Matrix[BR_Loc[0][i]:BR_Loc[0][i]+(Mesh_Complexity),BR_Loc[1][i]-1]=1
            Matrix[BR_Loc[0][i]+(Mesh_Complexity-2),BR_Loc[1][i]-(Mesh_Complexity-2)]=1
    else:
        for i in range(0,len(TL_Loc[0])):
            Matrix[TL_Loc[0][i]-1,TL_Loc[1][i]:TL_Loc[1][i]+Mesh_Complexity]=1
            Matrix[TL_Loc[0][i]-(Mesh_Complexity-1):TL_Loc[0][i],TL_Loc[1][i]+1]=1
            Matrix[TL_Loc[0][i]-(Mesh_Complexity-2),TL_Loc[1][i]+(Mesh_Complexity-2)]=1
            Matrix[TR_Loc[0][i]-1,TR_Loc[1][i]-(Mesh_Complexity-1):TR_Loc[1][i]]=1
            Matrix[TR_Loc[0][i]-(Mesh_Complexity-1):TR_Loc[0][i],TR_Loc[1][i]-1]=1
            Matrix[TR_Loc[0][i]-(Mesh_Complexity-2),TR_Loc[1][i]-(Mesh_Complexity-2)]=1
        for i in range(0,len(BL_Loc[0])):
            Matrix[BL_Loc[0][i]+1,BL_Loc[1][i]:BL_Loc[1][i]+Mesh_Complexity]=1
            Matrix[BL_Loc[0][i]:BL_Loc[0][i]+(Mesh_Complexity),BL_Loc[1][i]+1]=1
            Matrix[BL_Loc[0][i]+(Mesh_Complexity-2),BL_Loc[1][i]+(Mesh_Complexity-2)]=1
            Matrix[BR_Loc[0][i]+1,BR_Loc[1][i]-(Mesh_Complexity-1):BR_Loc[1][i]]=1
            Matrix[BR_Loc[0][i]:BR_Loc[0][i]+(Mesh_Complexity),BR_Loc[1][i]-1]=1
            Matrix[BR_Loc[0][i]+(Mesh_Complexity-2),BR_Loc[1][i]-(Mesh_Complexity-2)]=1
 
    return Matrix
    
    
    
            
            