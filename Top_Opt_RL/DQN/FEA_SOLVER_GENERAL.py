# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:23:01 2021

@author: nbrow
"""

import numpy as np

#import cupy
import skimage.measure as measure

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
    
    #U=cupy.matmul(cupy.linalg.inv(cupy.array(ActiveStiffness)),cupy.array(ActiveForce))
    #U=cupy.ndarray.get(U)
    U=np.matmul(np.linalg.inv(ActiveStiffness),ActiveForce)
    displacements=np.zeros((GDof,1))
    displacements[activeDof]=U
    return displacements
def stresses2D(GDof,TotElements,nodeCoor,numberNodes,displacements,UX,UY,materials,ScaleFactor,VoidCheck,elementNodes):    
        gaussLocations=np.array([[-.577350269189626, -.577350269189626],
                                 [ .577350269189626, -.577350269189626],
                                 [ .577350269189626,  .577350269189626],
                                 [-.577350269189626,  .577350269189626]])
        gaussWeights=np.array([[1],[1],[1],[1]])
        'stresses at nodes'
        stress=np.zeros((TotElements,len(elementNodes[0]),3))        
        for e in range(0,TotElements):
            indice=(elementNodes[e,:])
            indice=indice.astype(int)
            elementDof=[]
            elementDof=np.append(indice,indice+numberNodes)

            nn=len(indice)
            for q in range(0,len(gaussWeights)):
                pt=gaussLocations[q,:]
                xi=pt[0]
                eta=pt[1]
                sFQ4=shapeFunctionQ4(xi,eta)
                naturalDeriv=sFQ4[1]
                #Determine Jacobian
                JR=Jacobian(nodeCoor[indice,:],naturalDeriv)
                XYderivative=JR[2]
                'Create the B matrix'
                B=np.zeros((3,2*nn))
                B[0,0:nn]         =np.transpose(XYderivative[:,0])
                B[1,nn:(2*nn)]  =np.transpose(XYderivative[:,1])
                B[2,0:nn]         =np.transpose(XYderivative[:,1])
                B[2,nn:(2*nn)]  =np.transpose(XYderivative[:,0])

                'Element Deformation'
                #Check to See What Material Properties Should Be Assigned
                if VoidCheck[e]==1:
                    E=materials[0,0]
                    poisson=materials[0,1]
                else:
                    E=materials[1,0]
                    poisson=materials[1,1]
                C=np.matrix([[1,poisson,0], [poisson,1,0],[0,0,(1-poisson)/2]])*(E/(1-(poisson**2)))
                strain=np.matmul(B,displacements[elementDof])
                stress[e,q,:]=np.transpose(np.dot(C,strain))

        return(stress)
def FEASolve(VoidCheck,Lx,Ly,ElementsX,ElementsY,Loaded_Node,Loaded_Direction,BC1,BC2,Stress):
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
    P=5e5*Loaded_Direction#Magnitude
    
    'Mesh Generation'
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
    fixedNodeX=[BC1,BC2]
    fixedNodeY=[BC1+numberNodes,BC2+numberNodes]
 
    prescribedDof=[fixedNodeX+fixedNodeY]
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
    force[Loaded_Node]=P 
    FS2D=formStiffness2D(GDof,TotElements,numberNodes,elementNodes,nodeCoor,materials,1,1,VoidCheck)                                                         
    stiffness=FS2D[0]
    displacements=solution(GDof,prescribedDof,stiffness,force)
    UX=np.asarray(np.transpose(displacements[0:numberNodes]))
    UY=np.asarray(np.transpose(displacements[numberNodes:GDof]))
    MaxDisplacements=displacements.min()
    StrainEnergy=np.matmul(np.transpose(displacements),stiffness)
    StrainEnergy=np.matmul(StrainEnergy,displacements)
    RectOutput=rectangularmesh(Lx,Ly,ElementsX,ElementsY)

    if Stress is True:
        ScaleFactor=0.1      
        StressVal=stresses2D(GDof,TotElements,nodeCoor,numberNodes,displacements,UX,UY,materials,ScaleFactor,VoidCheck,elementNodes)
        stress=StressVal
        sigmax=stress[:,:,0]
        sigmay=stress[:,:,1]
        tauxy=stress[:,:,2]
    
        stressx=np.zeros((ElementsX+1,ElementsY+1))
        stressy=np.zeros((ElementsX+1,ElementsY+1))
        stressxy=np.zeros((ElementsX+1,ElementsY+1))
        #stressmat=np.zeros(numberNodes)
        Count=0
        
        for j in range(0,ElementsY):
            for i in range(0,ElementsX):
                #Change sigmax to sigmay for different stress distribution
               stressx[np.ix_(range(i,i+2),range(j,j+2))]=np.reshape(sigmax[Count,:],(2,2))
               stressy[np.ix_(range(i,i+2),range(j,j+2))]=np.reshape(sigmay[Count,:],(2,2))
               stressxy[np.ix_(range(i,i+2),range(j,j+2))]=np.reshape(tauxy[Count,:],(2,2))
               Count=Count+1
               
        ''' Calculate the Von Mises Stress'''
        VonMises=np.sqrt((stressx*stressx)-(stressx*stressy)+(stressy*stressy)+(3*stressxy*stressxy))
        VonMises_node=VonMises
        VonMises=VonMises[0:ElementsX,0:ElementsY]
        VonMises_node_nn=np.reshape(np.transpose(VonMises_node),(((ElementsX+1)*(ElementsY+1),1)))
        VonMises_nn=np.reshape(np.transpose(VonMises),((ElementsX*ElementsY),1))

        VonMises_node_nn=max(VonMises_node_nn)/VonMises_node_nn
        VonMises_node_nn[VonMises_node_nn>1e4]=0
        VonMises_node_nn=VonMises_node_nn/max(VonMises_node_nn)
        VonMises_nn=max(VonMises_nn)/VonMises_nn
        VonMises_nn[VonMises_nn>1e3]=0
        VonMises_nn=VonMises_nn/max(VonMises_nn)

        
    
    if Stress is False:
        VonMises=[]
        VonMises_nn=[]

    return(MaxDisplacements,StrainEnergy,VonMises,VonMises_nn,VonMises_node_nn)