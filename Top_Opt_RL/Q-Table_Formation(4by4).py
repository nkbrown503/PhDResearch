# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 13:15:12 2021

@author: nbrow
"""


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
Start=time.perf_counter()
'Input Linear Material Properties'


ElementsX=4 #Number of Elements in X Direction
ElementsY=4 #Number of Elements in Y Direction
TotElements=ElementsX*ElementsY #Total Number of Elements 

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



'Introduce the Voids to the Mesh'
#--------------------------------------------------------
#|                                                      |
#|                                                      |
#|  Randomly Introduce a Predetermined Number of Voids  |
#|                         |                            |
#|                         v                            |
#--------------------------------------------------------

Column_Name=[]
for i in range(0,TotElements):
    Column_Name.append(str(i))
    
state_table=pd.DataFrame([],columns=Column_Name)

VN=pd.DataFrame(list(np.transpose(np.ones((TotElements,1)))),columns=Column_Name)
state_table=state_table.append(VN)
LeftBC=list(range(0,ElementsX*(ElementsY-1)*2,ElementsX*(ElementsY-1)))
#LeftBC.append(ElementsX*(ElementsY)-1) #For Top Right
LeftBC.append(ElementsX-1) #For Bottom Right
for i in range(0,100000):
    iter=rand.randint(0,6)
    VoidCheck=np.ones((TotElements,1))
    FailCheck=0
    if len(state_table)>100000:
        break
    for j in range(0,iter+1):
        Clear=False
        while Clear!= True:
            removespot=rand.randint(0,TotElements-1)
                #Make sure the material being removed exists and its not the Top or Bottom Part of the Left BC
            if VoidCheck[removespot]==1 and removespot not in LeftBC:
                VoidCheck[removespot]=0
           
                #Check to ensure A Single Piece
                ElementMat=np.reshape(VoidCheck,(ElementsX,ElementsY))
                SingleCheck=isolate_largest_group_original(ElementMat)

                if SingleCheck[1]!=True or all(VoidCheck) in state_table:
                    FailCheck+=1
                    VoidCheck[removespot]=1
                    if FailCheck>=1000:
                        #print('Unable to Produce a Single Body')
                        break
                else:
                    VC_Post=pd.DataFrame(list(np.transpose(VoidCheck)),columns=Column_Name)
                    state_table=state_table.append(VC_Post)
                    Clear=True
                    FailCheck=0
                    if len(state_table)%5000 ==0: 
                        print('Current Size of State Table: '+ str(len(state_table)))
state_table=state_table.drop_duplicates()