# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:14:41 2022

@author: nbrow
"""


import numpy as np
from autoencoder import VAE
import os
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
import copy
from autoencoder import loss_func
import matplotlib.pyplot as plt 


def get_training_data(Use_Samples,Tot_Samples,y_size,x_size):
    x_train=np.zeros((Use_Samples,y_size,x_size))
    
    file_count =Use_Samples
    All_Trials=list(np.linspace(1,Tot_Samples,Tot_Samples).astype('int'))
    Train_Trials=random.sample(All_Trials,int(file_count*.90))
    All_Trials=[x for x in All_Trials if x not in Train_Trials]
    Test_Trials=random.sample(All_Trials, int(file_count*.10))
    x_train=np.zeros((len(Train_Trials),y_size,x_size))
    x_test=np.zeros((len(Test_Trials),y_size,x_size))
    for i in range(0,len(Train_Trials)):
        x_train[i,:,:]=np.load('../ML_Input_Files/UC_Design_{}.npy'.format(Train_Trials[i]))[0:20,0:60]
    for j in range(0,len(Test_Trials)):
        x_test[j,:,:]=np.load('../ML_Input_Files/UC_Design_{}.npy'.format(Test_Trials[j]))[0:20,0:60]
    x_train = np.reshape(x_train, newshape=(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)) 
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1))
    return x_train,x_test


def train(x_train,x_test,y_size,x_size,num_channels,latent_space_dim, learning_rate):
    autoencoder,encoder,decoder, encoder_mu,encoder_log_variance = VAE(y_size,x_size,num_channels,latent_space_dim)
    autoencoder.summary()
    autoencoder.compile(optimizer=Adam(lr=learning_rate), loss=loss_func(encoder_mu, encoder_log_variance))

    return autoencoder, encoder,decoder


if __name__ == "__main__":
    x_sample=np.load('../ML_Input_Files/UC_Design_1.npy')[0:20,0:60] #Import Lower Quarter of Unit-Cell
    y_size=np.shape(x_sample)[0]
    x_size=np.shape(x_sample)[1]
    num_channels=1
    Tot_Samples=110000
    Use_Samples=110000
    latent_space_dim=16
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 128
    EPOCHS = 150
    save_name='VAE_training_3-21-22'
    Method='Test'
    #x_train,x_test = get_training_data(Use_Samples,Tot_Samples,y_size,x_size)
    if Method=='Train':
        x_train=np.load('VAE_Training_Set.npy')
        x_test=np.load('VAE_Testing_Set.npy')
        autoencoder,encoder,decoder = train(x_train,x_test,y_size,x_size,num_channels,latent_space_dim,learning_rate=LEARNING_RATE)
        history=autoencoder.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_data=(x_test, x_test))
        plt.plot(history.history['loss'],label='Training Loss')
        plt.plot(history.history['val_loss'],label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        plt.savefig(save_name+'_LossPlot.png')
        autoencoder.save("model")
        encoder.save("VAE_encoder.h5") 
        decoder.save("VAE_decoder.h5")
    elif Method=='Test':
        x_test=np.load('VAE_Testing_Set.npy')
        autoencoder,encoder,decoder, _,_ = VAE(y_size,x_size,num_channels,latent_space_dim)
        encoder.load_weights("VAE_encoder_Best.h5") 
        decoder.load_weights("VAE_decoder_Best.h5")
        encoded_data = encoder.predict(x_test)
        
        decoded_data = decoder.predict(encoded_data)
        hold_dd=copy.deepcopy(decoded_data)
        decoded_data[decoded_data>0.4]=1
        decoded_data[decoded_data<=0.4]=0
        for i in range(0,3):
            x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
            Num=x[i]
            fig_1, axs_1 = plt.subplots()
            fig_2, axs_2 = plt.subplots()
            fig_3, axs_3 = plt.subplots()
            fig_4, axs_4 = plt.subplots()
            axs_1.imshow(x_test[Num,:,:,0],cmap='Blues',origin='lower')
            axs_2.imshow(decoded_data[Num,:,:,0],cmap='Blues',origin='lower')
            axs_3.imshow(hold_dd[Num,:,:,0],cmap='Blues',origin='lower')
            
            axs_4.plot(x,encoded_data[Num],'b.',markersize=20)
            axs_4.plot(x,encoded_data[Num],'r-')
            axs_4.set_xlim(1,16)
            axs_4.set_ylim(-1,1)
            
            axs_4.set_xlabel('Latent Variable')
            axs_4.set_ylabel('Value')