# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:42:39 2021

@author: nbrow
"""
import FEA_SOLVER_GENERAL
import numpy as np
import random
def LC_Nodes(Load_Element,Load_Direction,Lx,Ly,Elements_X,Elements_Y,Node_Location):
    '''Given the loaded element and loading direction, 
    produce the nodes that should be loaded for the FEA. If the user is testing
     and selects an element not on the exterior edges of the shape
     they will be prompted to select the top/right/bottom/left nodes 
     of the selected element '''
  

    Go_List,Elem_List,Bottom_List,Top_List,Left_List,Right_List=Element_Lists(Elements_X,Elements_Y)
    Load_Nodes=FEA_SOLVER_GENERAL.rectangularmesh(Lx,Ly,Elements_X,Elements_Y)[1][Load_Element,:]
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
        if Node_Location:
            Dir=int(input('Where would you like the load applied on the element (0:Bottom   1:Right  2:Top   3:Left): '))
        else:
            Dir=random.randrange(0,4)
        if Dir==0:
            Loaded_Node=Load_Nodes[0]
            Loaded_Node2=Load_Nodes[1]
        if Dir==1:
            Loaded_Node=Load_Nodes[1]
            Loaded_Node2=Load_Nodes[2]
        if Dir==2:
            Loaded_Node=Load_Nodes[2]
            Loaded_Node2=Load_Nodes[3]
        if Dir==3:
            Loaded_Node=Load_Nodes[3]
            Loaded_Node2=Load_Nodes[0]
    Loaded_Node=int(Loaded_Node)
    Loaded_Node2=int(Loaded_Node2)   
    return Loaded_Node, Loaded_Node2

def BC_Nodes(Load_Element,Lx,Ly,Elements_X,Elements_Y):
    
    ''''Given the Boundary Condition Element,produce the 
    corresponding nodes depending on where it's located'''
    
    _,_,Bottom_List,Top_List,Left_List,Right_List=Element_Lists(Elements_X,Elements_Y)
    Load_Nodes=FEA_SOLVER_GENERAL.rectangularmesh(Lx,Ly,Elements_X,Elements_Y)[1][Load_Element,:]
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
        Loaded_Node=Load_Nodes[2]
        Loaded_Node2=Load_Nodes[2]
    if Load_Element in Top_List and Load_Element in Left_List:
        Loaded_Node=Load_Nodes[3]
        Loaded_Node2=Load_Nodes[3]
    if Load_Element in Bottom_List and Load_Element in Right_List:
        Loaded_Node=Load_Nodes[1]
        Loaded_Node2=Load_Nodes[1]

    if Load_Element in Bottom_List and Load_Element in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[0]
    if Load_Element not in Bottom_List and Load_Element not in Top_List and Load_Element not in Right_List and Load_Element not in Left_List:
        Loaded_Node=Load_Nodes[0]
        Loaded_Node2=Load_Nodes[1]

    Loaded_Node=int(Loaded_Node)
    Loaded_Node2=int(Loaded_Node2)   
    return Loaded_Node, Loaded_Node2

def Element_Lists(Elements_X,Elements_Y):
    '''Simple function that produces a list of all the elements in the matrix '''
    Go_List=[]
    Elem_List=[]
    Go_List=np.append(Go_List,range(0,Elements_X+1))
    Elem_List=np.append(Elem_List,range(0,Elements_X))

    for num in range(0,Elements_Y-1):
        Go_List=np.append(Go_List,(Elements_X+1)*(num+1))
        Go_List=np.append(Go_List,(Elements_X+1)*(num+2)-1)
    Go_List=np.append(Go_List,range((Elements_X*(Elements_Y+1)),(Elements_X*(Elements_Y+2)+1)))
    for num in range(0,Elements_Y-2):
        Elem_List=np.append(Elem_List,(Elements_X*(num+1)))
        Elem_List=np.append(Elem_List,(Elements_X*(num+2)-1))
    Elem_List=np.append(Elem_List,range(Elements_X*(Elements_Y-1),(Elements_X*(Elements_Y))))
    Bottom_List=np.arange(0, Elements_X,1).tolist()
    Top_List=np.arange(Elements_X*(Elements_Y-1),Elements_X*Elements_Y, 1).tolist()
    Left_List=np.arange(0, Elements_X*(Elements_Y),Elements_X).tolist()
    Right_List=np.arange(Elements_X-1,Elements_X*Elements_Y+1,Elements_X).tolist()

    return Go_List, Elem_List, Bottom_List, Top_List,Left_List,Right_List