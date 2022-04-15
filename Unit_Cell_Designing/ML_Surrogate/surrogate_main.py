# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:00:52 2022

@author: nbrow
"""

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt 
from autoencoder import VAE
#from Resnet_model import create_res_net
from FCC_Surrogate_model import FCC_model

model = FCC_model() # or create_plain_net()

#model.summary()


save_name = 'UC_Surrogate_Model_Weights_4-5-22' # or 'cifar-10_plain_net_30-'+timestr
load_name = "UC_Surrogate_Model_Weights_4-4-22"
load_path="checkpoints/"+load_name+"/cp.ckpt"
checkpoint_path = "checkpoints/"+save_name+"/cp.ckpt"
Norm_Y=np.load('../Result_Files/Normalizing_Compression_Y_Value.npy')
Norm_X=np.load('../Result_Files/Normalizing_Compression_X_Value.npy')



# save model after each epoch
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1
)
#Segment Datasets 

file_count =100_000

UC_samp=np.load('../ML_Input_Noise_Files/UC_Design_1.npy')[0:20,0:60]
#Load in the VAE
_,encoder,_, _,_ = VAE(np.shape(UC_samp)[0],np.shape(UC_samp)[1],num_channels=1,latent_space_dim=16)

encoder.load_weights("../VAE/VAE_encoder_Best.h5") 

#------------------------------Comment Below-------------------------
'''
X_samp= encoder.predict(np.reshape(UC_samp,(1,20,60,1)))
Y_samp=np.load('../ML_Output_Noise_Files/UC_Design_C_1.npy')
X_All=np.zeros((file_count,X_samp.shape[0],X_samp.shape[1]))
Y_All=np.zeros((file_count,Y_samp.shape[0]))
for i in range(0,file_count):
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(i,file_count))
    sys.stdout.flush()  
    Coef=np.load('../ML_Output_Noise_Files/UC_Design_C_{}.npy'.format(i+1))
    UC_Train=np.load('../ML_Input_Noise_Files/UC_Design_{}.npy'.format(i+1))[0:20,0:60]

    X_All[i,:,:]=encoder.predict(np.reshape(UC_Train,(1,20,60,1)))
    Y_All[i,:]=Coef
np.save('X_All_Set.npy',X_All)
np.save('Y_All_Set.npy',Y_All)
X_All_Stdev=np.zeros((16,))
for i in range(0,np.shape(X_All)[2]):
    X_All_Stdev[i]=np.std(X_All[:,:,i])
np.save('X_All_Stdev.npy',X_All_Stdev)
'''
#--------------Comment Above---------------------------------


X_All=np.load('X_All_Set.npy')
Y_All=np.load('Y_All_Set.npy')
X_All_Stdev=np.load('X_All_Stdev.npy')

for i in range(0,np.shape(X_All)[2]):
    X_All[:,:,i]=X_All[:,:,i]/X_All_Stdev[i]
X_All=X_All.reshape(X_All.shape[0],X_All.shape[2])


tensorboard_callback = TensorBoard(
    log_dir='tensorboard_logs/'+save_name,
    histogram_freq=1)
Action='Train'
if Action=='Train': 
    history=model.fit(
        x=X_All,
        y=Y_All,
        epochs=150,
        verbose='auto',
        validation_split=.2,
        batch_size=128)
    model.save_weights(checkpoint_path)
    plt.plot(history.history['loss'],label='Training Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(save_name+'_LossPlot.png')

elif Action=='Test':
    model.load_weights(checkpoint_path)
    Tess=[1,2,3,4]
    ML_Tot=model.predict(X_All)
    for It in range(0,len(Tess)):
        ML_pred=ML_Tot[Tess[It]]
        Truth=Y_All[Tess[It]]
        print('\n-----------')
        print('Ground Truth')
        print(Truth)
        print('\nML Prediction')
        print(ML_pred)
elif Action=='Config':
    plt.figure()
    hist,bin_edges = np.histogram(Y_All,bins=100)
    plt.hist(Y_All,bins=bin_edges)
    plt.xlabel('Normalized Coef All')
    plt.ylabel('# of Values')

 

    
    
    
    
    
    
    