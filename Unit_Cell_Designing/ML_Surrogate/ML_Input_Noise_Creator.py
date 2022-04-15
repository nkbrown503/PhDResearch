# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:11:00 2022

@author: nbrow
"""
import numpy as np
import random
import sys
import matplotlib.pyplot as plt 
import copy 
from autoencoder import VAE
_,VAE_Encoder,_,_,_=VAE(20,60,1,16)
VAE_Encoder.load_weights("../VAE/VAE_encoder_Best.h5") 
UC_samp=np.load('../ML_Input_Noise_Files/UC_Design_1.npy')[0:20,0:60]
New_Count=92_568

for i in range(1,New_Count+1):
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(i,New_Count))
    sys.stdout.flush()

    UC_Num=random.randint(1,7432)
    True_UC=np.load('../ML_Input_Noise_Files/UC_Design_{}.npy'.format(UC_Num))
    Popt_Hold=np.load('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(UC_Num))
    
    Noise_Plot=copy.deepcopy(True_UC)
    X_loc=[random.randint(0,119) for i in range(120)]
    Y_loc=[random.randint(0,39) for i in range(120)]
    for Noise in range(0,len(X_loc)):
        if True_UC[Y_loc[Noise],X_loc[Noise]]==1:
            Noise_Plot[Y_loc[Noise],X_loc[Noise]]=0
        else:
            Noise_Plot[Y_loc[Noise],X_loc[Noise]]=1

    np.save('../ML_Input_Noise_Files/UC_Design_{}.npy'.format(i+7432),Noise_Plot)
    np.save('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(i+7432),Popt_Hold)