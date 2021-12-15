# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 19:57:01 2021

@author: nbrow
"""

import numpy as np
import sys 
import random
import math
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
def Bezier_3D_UC_Builder(E_X,E_Y,E_Z,Curves,Iterations,Mirror,Type):
    
    
    '''This function is used to build a unit-cell by sequentially adding material
    defined by Bezier Curves. The start and end points of the bezier curve are
    controlled by the 'Type' variable. 
    
    'Surface' will use the entire top and bottom 
    surfaces as potential starting and ending points for the bezier curve. A random point
    in the bottom surface will be used as the start point and a random point on the 
    top surface will be used as the ending point of the curve. 
    
    
    'Corner' will use the 8 cubic corners as potential starting and ending points for the 
    Bezier curve. The unit-cell build will not terminate until each corner has been used
    as a start or ending point at least once. This ensures that the build delivers a practical and 
    feasible unit-cell solutions.
    
    The 2 intermediate points of the Bezier curve are randomly select as any point between the starting and 
    end points.
    '''
    
    
    
    
    if Type!='Surface' and Type!='Corner':
        sys.exit('Type must be either `Surface` or `Corner`... please check for a misspelling')
    Element_Block=np.zeros((E_X,E_Y,E_Z))
    
    if Type=='Surface':
       Element_Block[0:E_X,0:E_Y,0]=1
       if not Mirror:
           Element_Block[0:E_X,0:E_Y,E_Z]=1
            
    lm=1
    if Type=='Surface':
        Clear= True 
    else:
        Clear=False
    while lm<=Curves or not Clear:
     
        if Type=='Surface':
            P_0=[random.randint(0,E_X-1),random.randint(0,E_Y-1),0]
            P_3=[random.randint(0,E_X-1),random.randint(0,E_Y-1),E_Z-1]
        elif Type=='Corner':
            P_0=[random.choice([0,E_X-1]),0,random.choice([0,E_Z-1])]
            P_3=[random.choice([0,E_X-1]),E_Y-1,random.choice([0,E_Z-1])]
 


        P_1=[random.randint(0,E_X-1),random.randint(0,E_Y-1),random.randint(0,E_Z-1)]
        P_2=[random.randint(0,E_X-1),random.randint(0,E_Y-1),random.randint(0,E_Z-1)]
    

        Thickness=random.choice([1,1])
        
        P_0_aug=[P_0[0]/E_X,P_0[1]/E_Y,P_0[2]/E_Z]
        P_1_aug=[P_1[0]/E_X,P_1[1]/E_Y,P_1[2]/E_Z]
        P_2_aug=[P_2[0]/E_X,P_2[1]/E_Y,P_2[2]/E_Z]
        P_3_aug=[P_3[0]/E_X,P_3[1]/E_Y,P_3[2]/E_Z]
        t=np.arange(0, 1.01, 0.01).tolist()
        B_X=np.zeros((len(t),1))
        B_Y=np.zeros((len(t),1))
        B_Z=np.zeros((len(t),1))
        
        for i in range(0,len(t)):
            B_X[i]=(((1-t[i])**3)*P_0_aug[0])+(3*((1-t[i])**2)*t[i]*P_1_aug[0])+(3*(1-t[i])*t[i]**2*P_2_aug[0])+(t[i]**3*P_3_aug[0])
            B_Y[i]=(((1-t[i])**3)*P_0_aug[1])+(3*((1-t[i])**2)*t[i]*P_1_aug[1])+(3*(1-t[i])*t[i]**2*P_2_aug[1])+(t[i]**3*P_3_aug[1])
            B_Z[i]=(((1-t[i])**3)*P_0_aug[2])+(3*((1-t[i])**2)*t[i]*P_1_aug[2])+(3*(1-t[i])*t[i]**2*P_2_aug[2])+(t[i]**3*P_3_aug[2])
        
        X_Seg=np.arange(0,1,1/(E_X+1))
        Y_Seg=np.arange(0,1,1/(E_Y+1))
        Z_Seg=np.arange(0,1,1/(E_Z+1))

        Seg_Nodes_X=np.zeros((E_X,2))
        Seg_Nodes_Y=np.zeros((E_Y,2))
        Seg_Nodes_Z=np.zeros((E_Z,2))
        
        for i in range(0,E_X):
            Seg_Nodes_X[i,0]=X_Seg[i]
            Seg_Nodes_X[i,1]=X_Seg[i+1]
            
        for j in range(0,E_Y):
            Seg_Nodes_Y[j,0]=Y_Seg[j]
            Seg_Nodes_Y[j,1]=Y_Seg[j+1]
        for k in range(0,E_Z):
            Seg_Nodes_Z[k,0]=Z_Seg[k]
            Seg_Nodes_Z[k,1]=Z_Seg[k+1]
                    
        

        for k in range(0,len(B_X)-1):
            X_Test=(Seg_Nodes_X[:,0]<=B_X[k]) & (Seg_Nodes_X[:,1]>B_X[k])
            Y_Test=(Seg_Nodes_Y[:,0]<=B_Y[k]) & (Seg_Nodes_Y[:,1]>B_Y[k])
            Z_Test=(Seg_Nodes_Z[:,0]<=B_Z[k]) & (Seg_Nodes_Z[:,1]>B_Z[k])
            Remove_X=int(np.where(X_Test==True)[0][0])
            Remove_Y=int(np.where(Y_Test==True)[0][0])
            Remove_Z=int(np.where(Z_Test==True)[0][0])

            Loc_X=math.modf(Remove_X/(E_X))
            Loc_Y=math.modf(Remove_Y/(E_Y))
            Loc_Z=math.modf(Remove_Z/(E_Z))
            Upper_Bound=int(min(int((E_Y)-(Loc_Y[1]+0.001)),Thickness))
            Lower_Bound=int(min(int(Loc_Y[1]+0.001),Thickness))
            Left_Bound=int(min(int((E_X)*Loc_X[0]),Thickness))
            Right_Bound=int(min(int((E_X)-((E_Y)*Loc_X[0])),Thickness))
            Back_Bound=int(min(int(Loc_Z[1]+0.001),Thickness))
            Front_Bound=int(min(int((E_Z)-(Loc_Z[1]+0.001)),Thickness))
        
            Element_Block[Remove_X-Left_Bound:Remove_X+Right_Bound,Remove_Y-Lower_Bound:Remove_Y+Upper_Bound,Remove_Z-Back_Bound:Remove_Z+Front_Bound]=1


        
        if Type=='Corner' and Element_Block[0,0,0]==1 and Element_Block[E_X-1,0,0]==1 and Element_Block[0,E_Y-1,0]==1 and Element_Block[0,0,E_Z-1]==1 and Element_Block[E_X-1,E_Y-1,0]==1 and Element_Block[0,E_Y-1,E_Z-1]==1 and Element_Block[E_X-1,0,E_Z-1]==1 and Element_Block[E_X-1,E_Y-1,E_Z-1]==1:
            Clear=True
        lm+=1
    return(Element_Block)

def Mirror_Func(E_X,E_Y,E_Z,Element_Block,It,Mirror,Print_UC,Save,Tesselate):
    '''This function will mirror the unit-cell designed by the above function about the 
    x,y, and z axes to promote symmettry if this unit-cell were to be loaded. 
    If a user does not wish to tesselate the unit-cell then the unit-cell design will be 
    saves in the 'Design_Files' folder. If the user does want to tesselate the unit-cell 
    the saving proceudre is accounted for in the next function.
    
    
    
    This function will also print the images of the isometric, front, top, and right views 
    of the unit-cell. Please note the images are produces using a scatter plot which gives an acceptable
    representation of the unit-cell but alternative methods could be implemented to improve the visability.
    Regardless, implementing the unit-cells in a design or FEA fashion can be completed 
    using the numpy file.'''
    if Mirror:
        Element_Plot=np.zeros((E_X*2,E_Y*2,E_Z*2))
        Element_M1=np.zeros((E_X*2,E_Y,E_Z)) #Mirror about X axis 
        Element_M1[0:E_X,0:E_Y,0:E_Z]=Element_Block
        Element_M1[E_X:E_X*2,0:E_Y,0:E_Z]=np.flip(Element_Block,axis=0)
        Element_M2=np.zeros((E_X*2,E_Y,E_Z*2)) #Mirror about Z Axis
        Element_M2[0:E_X*2,0:E_Y,0:E_Z]=Element_M1
        Element_M2[0:E_X*2,0:E_Y,E_Z:E_Z*2]=np.flip(Element_M1,axis=2)
        Element_Plot[0:E_X*2,0:E_Y,0:E_Z*2]=Element_M2 #Mirror about Y Axis
        Element_Plot[0:E_X*2,E_Y:E_Y*2,0:E_Z*2]=np.flip(Element_M2,axis=1)
    if Print_UC:
        Plot_Loc_Top=np.where(Element_Plot[0:Element_Plot.shape[0],0:Element_Plot.shape[1],1:Element_Plot.shape[2]-1]==1)  
        Plot_Loc=np.where(Element_Plot[0:Element_Plot.shape[0],0:Element_Plot.shape[1],0:Element_Plot.shape[2]]==1)  
        fig = plt.figure()
        
        ax_iso = fig.add_subplot(221, projection='3d')
        ax_iso.scatter(Plot_Loc[0],Plot_Loc[1],Plot_Loc[2],marker=",",color='grey',edgecolors='black',s=50)
        ax_iso.view_init(azim=45,elev=45)
        ax_iso.axis('off')
        ax_YZPlane = fig.add_subplot(222, projection='3d')
        ax_YZPlane.scatter(Plot_Loc[0],Plot_Loc[1],Plot_Loc[2],marker=",",color='grey',edgecolors='black',s=5)
        ax_YZPlane.view_init(azim=0,elev=0)
        ax_XYPlane = fig.add_subplot(223, projection='3d')
        ax_XYPlane.scatter(Plot_Loc_Top[0],Plot_Loc_Top[1],Plot_Loc_Top[2],marker=",",color='grey',edgecolors='black',s=5)
        ax_XYPlane.view_init(azim=0,elev=90)
        ax_XZPlane = fig.add_subplot(224, projection='3d')
        ax_XZPlane.scatter(Plot_Loc[0],Plot_Loc[1],Plot_Loc[2],marker=",",color='grey',edgecolors='black',s=5)
        ax_XZPlane.view_init(azim=90,elev=0)
        ax_iso.set_title('Isometric',loc='center')
        ax_YZPlane.set_title('Front',loc='center')
        ax_XYPlane.set_title('Top (Top Layer Removed)',loc='center')
        ax_XZPlane.set_title('Right',loc='center')
        plt.show()
    if Save and not Tesselate:
        np.save('Design_Files/UC_Design_Image_{}.npy'.format(It),Element_Plot)
    return Element_Plot
    
def Tesselate_UC(E_X,E_Y,E_Z,UX,UY,UZ,It,Mirror_UC,Type,Save,Tesselate,Mirror):
    
    '''This function will tesselate the design unit-cell in the X,Y, and Z directions according to
    the variables UX, UY, UZ, which can be altered in the 'opts' file. When 'Type'=Surface, The tesselatation will introduce a solid 
    top and bottom material layers to ensure the tesselated design is a single body. This additional step is not 
    included when 'Type'= Corner as the tesselated designed is already guaranteed to be a single body. Adding the solid 
    top and bottom material surfaces could very easily be implemented if need by following the procedure shown
    below under the conditional 'If Tesselate and Type=Surface'
    
    Due to single body constraints, a unit-cell can only be tesselated if it has been mirrored. Future generations may adjust this constraint.
    
    Finally, if the user has decided to tesselate, the final tesselation will be saved in the 'Design_Files' folder '''
    
    
    
    if not Mirror:
        sys.exit('You can only tesselate a unitcell if it is mirrored...')
    if Tesselate and Type=='Corner':
        Tesselated_Design=np.zeros((E_X*2*UX,E_Y*2*UY,(E_Z*2*UZ)))
        for z_move in range(0,UZ):
            for y_move in range(0,UY):
                for x_move in range(0,UX):
                    Tesselated_Design[(E_X*2)*x_move:(E_X*2)*(x_move+1),(E_Y*2)*(y_move):(E_Y*2)*(y_move+1),(E_Z*2)*(z_move):(E_Z*2)*(z_move+1)]=Mirror_UC
    if Tesselate and Type=='Surface':
        Tesselated_Design_Hold=np.zeros((E_X*2*UX,E_Y*2*UY,(E_Z*2*UZ)-(UZ-1)))
        for z_move in range(0,UZ):
            for y_move in range(0,UY):
                for x_move in range(0,UX):
                    Tesselated_Design_Hold[(E_X*2)*x_move:(E_X*2)*(x_move+1),(E_Y*2)*(y_move):(E_Y*2)*(y_move+1),(E_Z*2)*(z_move):((E_Z*2)*(z_move+1))-2]=Mirror_UC[0:E_X*2,0:E_Y*2,1:(E_Z*2)-1]
        Tesselated_Design=np.zeros((E_X*2*UX,E_Y*2*UY,E_Z*2*UZ))
        Tesselated_Design[0:E_X*2*UX,0:E_Y*2*UY,1:E_Z*2*UZ-1]=Tesselated_Design_Hold
        Tesselated_Design[0:E_X*2*UX,0:E_Y*2*UY,0]=1
        Tesselated_Design[0:E_X*2*UX,0:E_Y*2*UY,(E_Z*2*UZ)-1]=1
    
    if Tesselate and Save:
        np.save('Design_Files/UC_Design_Tesselate{}by{}by{}_Image_{}.npy'.format(UX,UY,UZ,It),Tesselated_Design)
       
    
    
    
    
    
    
    
