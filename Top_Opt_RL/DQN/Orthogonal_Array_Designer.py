# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 11:07:49 2021

@author: nbrow
"""
import numpy as np
import pandas as pd
import math
from Top_Opt_RL.DQN. Node_Element_Extraction import Element_Lists
Elements_X=20
Elements_Y=20
Go_List,Elem_List,Bottom_List,Top_List,Left_List,Right_List=Element_Lists(Elements_X,Elements_Y)
Tot_List=[]
Tot_List=np.append(Tot_List,range(0,Elements_X*Elements_Y))
BC1E=Elem_List
BC2E=Elem_List
LC=Tot_List
Type=[0,1]
Ortho_Array=pd.DataFrame(columns=['BC1E','BC2E','LC','Type'])
for k in range(0,len(Type)):
    for i in range(0,len(LC)):
        for j in range(0,len(BC1E)):
            LC_It=LC[i]
            BC1_It=BC1E[j]
            if j+i<len(BC2E):
                BC2_It=BC2E[j+i]
            else:
                BC2_It=BC2E[math.ceil(math.modf((j+i)/len(BC2E))[0]*len(BC2E))]

            Ortho_Array=Ortho_Array.append({'BC1E': BC1_It, 'BC2E': BC2_It,'LC': LC_It,'Type':Type[k]},ignore_index=True)
Ortho_Array=Ortho_Array[Ortho_Array['BC1E']!=Ortho_Array['LC']]
Ortho_Array=Ortho_Array[Ortho_Array['BC2E']!=Ortho_Array['LC']]
Ortho_Array.to_pickle('Trial_Data/Orthogonal_Testing_Array_'+str(Elements_X)+'by'+str(Elements_Y)+'.pkl')

