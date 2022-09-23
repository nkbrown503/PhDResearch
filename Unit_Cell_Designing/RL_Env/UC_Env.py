# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 08:33:14 2022

@author: nbrow
"""


import numpy as np
from gym import Env
import copy 
from gym.spaces import Discrete, Box
import random
import sys
sys.path.insert(0,r'C:\Users\nbrow\OneDrive - Clemson University\Classwork\Doctorate Research\Python Coding\Unit_Cell_Design\VAE')
from autoencoder import VAE, Autoencoder
from RL_Bezier import RL_Bezier_Design as RLBD
from Matrix_Transforms import isolate_largest_group_original
from FCC_Surrogate_model import FCC_model
import math
import matplotlib.pyplot as plt 


class UC_Env(Env):
    def __init__(self,Type):
        #Action Space [Starting Corner, Ending Corner, Y Coor of IP1, X Coor of IP2, Y Coor of IP2, X Coor of IP2, Thickness]
        self.action_space=Box(np.array([0,0,0,0,0,0,0]),np.array([1,1,1,1,1,1,1]))
        self.max_steps=6
        self.E_Y=20
        self.E_X=60
        self.Legal=False
        self.Type=Type
        
        #The following values are used to standardize the FD curves and/or the Latent Space 
        if self.Type=='Compression':
            self.Force_Mean=np.load('../Constant_Values/Force_Mean_Values.npy')
            self.Force_stdev=np.load('../Constant_Values/Force_stdev_Values.npy')
            self.Latent_Space_SD=np.load('../Constant_Values/X_All_Compression_Stdev.npy')
        elif self.Type=='Tension':
            self.Force_Mean=np.load('../Constant_Values/Force_Mean_Tension_Values.npy')
            self.Force_stdev=np.load('../Constant_Values/Force_stdev_Tension_Values.npy')
            self.Latent_Space_SD=np.load('../Constant_Values/X_All_Tension_Stdev.npy')
        
        #Import the Autoencoder needed to define the latent space of each unit cell
        _,self.AE_Encoder,_=Autoencoder(self.E_Y,self.E_X,num_channels=1,latent_space_dim=48)
        self.AE_Encoder.load_weights("../VAE/AE_encoder_48.h5") 
        
        #Import the surrogate model to predict the FD curve of each unit cell 
        self.surrogate_model= FCC_model() 
        if self.Type=='Compression':
            self.surrogate_model.load_weights('../Surrogate_Model/checkpoints/UC_Compression_Surrogate_Model_Weights_9-6-22/cp.ckpt')
        elif self.Type=='Tension':
            self.surrogate_model.load_weights('../Surrogate_Model/checkpoints/UC_Tension_Surrogate_Model_Weights_9-5-22/cp.ckpt')


    def step(self,action):

        self.state_UC_=copy.deepcopy(self.state_UC)
        
        #Add material in the shape of a Bezier curve design according to the action from the RL agent 
        self.state_UC=RLBD(action,self.state_UC)
        try:
            self.Current_Force_=copy.deepcopy(self.Current_Force)
        except:
                'Nothing'
        self.obs_=copy.deepcopy(self.obs)
        
        #Take the unit cell and produce the 48 dimensional latent space 
        self.obs_[11:]=np.reshape(self.AE_Encoder.predict(np.reshape(self.state_UC,(1,20,60,1)),verbose=0)/self.Latent_Space_SD,(48,))
        self.step_count+=1
        SingleCheck=isolate_largest_group_original(self.state_UC)

        #Check if the unit cell is legal
        if SingleCheck[1]==False and self.step_count>1:
            Reward=-1
            Done=False
            self.Legal=False
            self.Perc_Error=1
            self.Force_Error=1
        elif SingleCheck[1]==False and self.step_count==1:
            Reward=0
            self.Force_Error=1
            Done=False
            self.Legal=False
            self.Perc_Error=1
            
        else:
            self.Legal=True
            self.state_UC=np.reshape(self.state_UC,(20,60))
            
            #Predict the FD curve accordinging to the latent space of the unit cell
            self.Current_Force=self.surrogate_model.predict(np.reshape(self.obs_[11:],(1,48)),verbose=0)[0]
            #self.Current_Force[1:]=(self.Current_Force[1:]*self.Force_stdev[1:])+self.Force_Mean[1:]
            
            #Unstandardize the FD curves
            self.Current_Force_Plot=copy.deepcopy(self.Current_Force)
            self.Current_Force_Plot[1:]=(self.Current_Force_Plot[1:]*self.Force_stdev[1:])+self.Force_Mean[1:]
            
            #Compare the error between the Desired FD and the True FD 
            self.Force_Error=(np.max([abs((i-j)) for i,j in zip(self.Current_Force_Plot[1:],self.Desired_Force_Plot[1:])]))
            self.Perc_Error=(np.max([abs((i-j)/j) for i,j in zip(self.Current_Force_Plot[4:],self.Desired_Force_Plot[4:])]))
            self.Perc_Error2=(np.max([abs((i-j)/i) for i,j in zip(self.Current_Force_Plot[4:],self.Desired_Force_Plot[4:])]))
            

            Reward=np.max([-self.Perc_Error,-self.Perc_Error2,-1])

            if self.Perc_Error<0.1 or self.Perc_Error2<0.1:
                #If the percent error is less than 10% than the design is considered satisfactory 
                Done=True
            else: 
                Done=False

        if self.step_count>=self.max_steps:
            Done=True 

        self.obs=copy.deepcopy(self.obs_)
        return self.obs_, Reward, Done, self.Legal
            
    def render(self,Legal,i):
        
        #Reformat 20x60 design domain into the 40x120 unit cell for plotting 
        self.state_UC=np.reshape(self.state_UC,(20,60))
        self.EX=np.shape(self.state_UC)[1]
        self.EY=np.shape(self.state_UC)[0]
        self.Element_Plot=np.zeros((self.EY*2,self.EX*2))
        self.Element_Plot[0:self.EY,0:self.EX]=self.state_UC
        self.Element_Plot[0:self.EY,self.EX:2*self.EX]=np.flip(self.state_UC,axis=1)
        self.Element_Plot[self.EY:self.EY*2,0:self.EX]=np.flip(self.state_UC,axis=0)
        self.Element_Plot[self.EY:self.EY*2,self.EX:2*self.EX]=np.flip(np.flip(self.state_UC,axis=0),axis=1)
        self.Strain_val=np.linspace(0,1,11)
        
        #print the proposed unit cell 
        fig,ax= plt.subplots()
        ax.imshow(self.Element_Plot,cmap='Blues',origin='lower')
        ax.axis('off')

        fig2,ax2=plt.subplots()
        C=['#8B7765','#CD3333','#473C8B','#6CA6CD','#CD4F39','#458B74','#CD69C9','#8E8E38']
        ax2.plot(self.Strain_val,self.Desired_Force_Plot,'--',color='{}'.format(C[i]),label='Desired Response')
        
        if self.Legal:
            
            self.Current_Force_Plot=copy.deepcopy(self.Current_Force)
            
            self.Current_Force_Plot[1:]=(self.Current_Force_Plot[1:]*self.Force_stdev[1:])+self.Force_Mean[1:]
            #Print a comparison between the desired and current FD curves 
            ax2.plot(self.Strain_val,self.Current_Force_Plot,'-',color='{}'.format(C[i]),label='Current Response')
            #ax2.legend()
            ax2.set_xlim([0,1])
            ax2.set_ylim([0,1])
            ax2.set_xlabel('Normalized Displacement')
            ax2.set_ylabel('Normalized Force')
            ax2.set_title('Force-Displacement Curve Comparison')
            
        
    def reset(self,Test,Type,i):
        
        self.step_count=0
        #Reset the unit cell and the RL observation
        self.state_UC=np.zeros((20,60))
        self.state_UC[0,0]=1
        self.state_UC[-1,0]=1
        self.state_UC[0,-1]=1
        self.state_UC[-1,-1]=1
        self.obs=np.zeros((59,)) #Top 11 Values are desired force response and bottom 48 are current latent space

        Small=True
        while Small==True:
            #Depending on the load type, randomly pick a unit cell to try to design
            if self.Type=='Compression':
                self.UC=random.randint(1,6000)
                #self.UC=random.choice(np.load('../Surrogate_Model/RL_Training_Set_C.npy'))
            else:
                self.UC=random.randint(1,3100)
            #----------Test cases----------------------------
            #self.UC=[1751,2433,1444,2402,941,1132,5843] #Softening
            #self.UC=[4115,985,4468] #stiffening
            #self.UC=[2018,2749,1047,1837,4883,661] #Linear
            #self.UC=[4670]#Inflection
            #self.UC=[1030,1852,245,1104,3069,2397] #Tension
            #Gold Red Purple Blue Orange Green
            #---------------------------------------------------
                
            #Get the FD curve depending on the desired unit cell 
            if Type=='Compression':
                self.Desired_Force=np.load('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(self.UC))
            elif Type=='Tension':
                self.Desired_Force=np.load('../ML_Output_Noise_Files/UC_Design_T_{}.npy'.format(self.UC))


            if Test==False:
                #Add noise to the desired FD curve during training 
                self.Desired_Force[1:]+=np.random.uniform(low=-.1,high=.1,size=10)
            #Unstandardize the FD curve 
            self.Desired_Force_Plot=copy.deepcopy(self.Desired_Force)
            self.Desired_Force_Plot[1:]=(self.Desired_Force_Plot[1:]*self.Force_stdev[1:])+self.Force_Mean[1:]

            if self.Desired_Force_Plot[-1]>0.1 or Test==True:
                #Training yields better results with larger values
                Small=False
        self.obs[:11]=np.reshape(self.Desired_Force,(11,))

        return self.obs
    
        
