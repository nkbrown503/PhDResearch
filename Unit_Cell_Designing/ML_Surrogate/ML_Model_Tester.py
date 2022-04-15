# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 19:19:53 2022

@author: nbrow
"""
from Resnet_model import create_res_net
from FCC_Surrogate_model import FCC_model
import matplotlib.pyplot as plt 
import numpy as np 
import sys
from autoencoder import VAE
import math
import random

surrogate_model= FCC_model() # or create_plain_nest()
_,VAE_Encoder,_,_,_=VAE(20,60,1,16)
VAE_Encoder.load_weights("../VAE/VAE_encoder_Best.h5") 
surrogate_model.load_weights('checkpoints/UC_Surrogate_Model_Weights_4-5-22\cp.ckpt')

Tess=np.load('../Testing_Sample_Numbers_3-1-22.npy')
Plot=False
avg_error=[]
It=1000
Norm_Y=np.load('../Result_Files/Normalizing_Compression_Y_Value.npy')
Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')


pred_list=[]
true_list=[]
pl1=[]
pl2=[]
pl3=[]
pl4=[]
p1_real=[]
p2_real=[]
p3_real=[]
p4_real=[]
Good=[]
Colors=['tab:blue','tab:orange','tab:green','tab:red','tab:cyan']
X_All_Stdev=np.load('X_All_Stdev.npy')
Good=[4617,6619,2462,6434,869]
for i in range(0,It):
    Val=random.randint(1,100_000)

    

    FileName_C='UC_Design_AR3_C_Trial{}'.format(int(Val))

    ML_input=np.load('../ML_Input_Noise_Files/UC_Design_{}.npy'.format(Val))[0:20,0:60]
    True_Force=np.load('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(Val))
    ML_input=ML_input.reshape(1,20,60,1)
    
    ML_input=VAE_Encoder.predict(ML_input)/X_All_Stdev
    Pred_Force=surrogate_model.predict(np.reshape(ML_input,(1,16)))[0]

    Force_Mean=np.load('../Force_Mean_Values.npy')
    Force_stdev=np.load('../Force_stdev_Values.npy')

    True_Force[1:]=(True_Force[1:]*Force_stdev[1:])+Force_Mean[1:]
    Pred_Force[1:]=(Pred_Force[1:]*Force_stdev[1:])+Force_Mean[1:]
    Pred_Force[0]=0
    if np.min(Pred_Force)>=0 and np.min(True_Force)>=0 and np.max(True_Force)<=1:
        pred_list=np.append(pred_list,Pred_Force)
        true_list=np.append(true_list,True_Force)
    Strain_val=np.linspace(0,1,11)
 
    #Pred_Fit_C=np.array([(Strain**3*Pred_Coef[0])+(Strain**2*Pred_Coef[1])+(Strain*Pred_Coef[2])+Pred_Coef[3] for Strain in Strain_val]) 
    #True_Fit_C=np.array([(Strain**3*True_Coef[0])+(Strain**2*True_Coef[1])+(Strain*True_Coef[2])+True_Coef[3] for Strain in Strain_val]) 

    c = np.mean([abs((i-j)/i)*100 for i,j in zip(True_Force[7:len(True_Force)],Pred_Force[7:len(Pred_Force)])])
    if c<50:
        avg_error=np.append(avg_error,c)
    #if c<=5:
    #    print(len(Good))
    #    Good=np.append(Good,Val)
    
    if Plot:
        plt.plot(Strain_val,Pred_Force,'-',color=Colors[i],label='True: UC {}'.format(Val))
        plt.plot(Strain_val,True_Force,'--',color=Colors[i],label='Prediction: UC {}'.format(Val))
        
        plt.legend(loc='best',prop={'size': 8})
        plt.xlabel('Normalized Displacement')
        plt.ylabel('Normalized Force')


print('\nThe average mean % error is: {}%'.format(abs(np.mean(avg_error))))


#plt.scatter(pred_list,true_list)
#plt.xlabel('Real Normalized Force Value')
#plt.ylabel('Predicted Normalized Force Value')


