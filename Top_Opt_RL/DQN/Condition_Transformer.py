# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:19:59 2021

@author: nbrow
"""
import numpy as np
import math
import FEA_SOLVER_GENERAL
def BC_Nodes(BC_Element,Old_EX,Old_EY):
    BC_Element=BC_Element
    ElementNodes=FEA_SOLVER_GENERAL.rectangularmesh(1,1,Old_EX,Old_EY)[1]
    Bottom_List=np.arange(0, Old_EX,1).tolist()
    Top_List=np.arange(Old_EX*(Old_EY-1),Old_EX*Old_EY, 1).tolist()
    Left_List=np.arange(0, Old_EX*(Old_EY),Old_EX).tolist()
    Right_List=np.arange(Old_EX-1,Old_EX*Old_EY+1,Old_EX).tolist()
    Load_Nodes=ElementNodes[BC_Element,:]
    if BC_Element in Bottom_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]

    if BC_Element in Top_List:
        Loaded_Node=Load_Nodes[2]
        Loaded_Node2=Load_Nodes[3]

    if BC_Element in Right_List:
        Loaded_Node=Load_Nodes[1]
        Loaded_Node2=Load_Nodes[2]

    if BC_Element in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[3]

    if BC_Element in Top_List and BC_Element in Right_List:
        Loaded_Node=Load_Nodes[2]
        Loaded_Node2=Load_Nodes[2]

    if BC_Element in Top_List and BC_Element in Left_List:
        Loaded_Node=Load_Nodes[3]
        Loaded_Node2=Load_Nodes[3]

    if BC_Element in Bottom_List and BC_Element in Right_List:
        Loaded_Node=Load_Nodes[1]
        Loaded_Node2=Load_Nodes[1]

    if BC_Element in Bottom_List and BC_Element in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[0]
  
    if BC_Element not in Bottom_List and BC_Element not in Top_List and BC_Element not in Right_List and BC_Element not in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]

    Loaded_Node=int(Loaded_Node)
    Loaded_Node2=int(Loaded_Node2)  
    return Loaded_Node, Loaded_Node2

def LC_Nodes(Load_Element,Load_Direction,Old_EX,Old_EY):
    ElementNodes=FEA_SOLVER_GENERAL.rectangularmesh(1,1,Old_EX,Old_EY)[1]
    Bottom_List=np.arange(0, Old_EX,1).tolist()
    Top_List=np.arange(Old_EX*(Old_EY-1),Old_EX*Old_EY, 1).tolist()
    Left_List=np.arange(0, Old_EX*(Old_EY),Old_EX).tolist()
    Right_List=np.arange(Old_EX-1,Old_EX*Old_EY+1,Old_EX).tolist()
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
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]
    Loaded_Node=int(Loaded_Node)
    Loaded_Node2=int(Loaded_Node2)   
    return Loaded_Node, Loaded_Node2

def Condition_Transform(Old_EX,Old_EY,New_EX,New_EY,BC1,BC2,BC1_E,BC2_E,LC,Load_Direction):
    
    Row_LC=math.floor(LC/New_EY)
    Col_LC=math.floor(round(math.modf(LC/New_EX)[0],1)*New_EX)
    Row_BC1_E=math.floor(BC1_E/New_EY)
    Col_BC1_E=math.floor(round(math.modf(BC1_E/New_EX)[0],1)*New_EX)
    Row_BC2_E=math.floor(BC2_E/New_EY)
    Col_BC2_E=math.floor(round(math.modf(BC2_E/New_EX)[0],1)*New_EX)
    Old_X_Perc=Row_LC/New_EY
    Old_Y_Perc=Col_LC/New_EX
    Old_X_Perc_BC1_E=Row_BC1_E/New_EY
    Old_Y_Perc_BC1_E=Col_BC1_E/New_EX
    Old_X_Perc_BC2_E=Row_BC2_E/New_EY
    Old_Y_Perc_BC2_E=Col_BC2_E/New_EX
    New_Row_LC=math.floor(Old_X_Perc*Old_EX)
    New_Col_LC=math.floor(Old_Y_Perc*Old_EY)
    New_Row_BC1_E=math.floor(Old_X_Perc_BC1_E*Old_EX)
    New_Col_BC1_E=math.floor(Old_Y_Perc_BC1_E*Old_EY)
    New_Row_BC2_E=math.floor(Old_X_Perc_BC2_E*Old_EX)
    New_Col_BC2_E=math.floor(Old_Y_Perc_BC2_E*Old_EY)
    New_LC_E=(New_Row_LC*Old_EX)+New_Col_LC
    New_BC1_E=(New_Row_BC1_E*Old_EX)+New_Col_BC1_E
    New_BC2_E=(New_Row_BC2_E*Old_EX)+New_Col_BC2_E
    New_LC1,New_LC2=LC_Nodes(New_LC_E,Load_Direction,Old_EX,Old_EY)
    New_BC1,New_BC2=BC_Nodes(New_BC1_E,Old_EX,Old_EY)
    New_BC3,New_BC4=BC_Nodes(New_BC2_E,Old_EX,Old_EY)
  

    return New_BC1, New_BC2,New_BC3,New_BC4,New_BC1_E,New_BC2_E,New_LC_E,New_LC1,New_LC2

            
            
            
