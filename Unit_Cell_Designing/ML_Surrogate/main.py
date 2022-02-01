# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 15:00:52 2022

@author: nbrow
"""

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

import os
import numpy as np
import random
import matplotlib.pyplot as plt 

from Resnet_model import create_res_net
from simple_model import simple_model

model = create_res_net() # or create_plain_net()
#model.summary()


name = 'UC_Surrogate_Model_Weights_UpdatedRelu' # or 'cifar-10_plain_net_30-'+timestr

checkpoint_path = "checkpoints/"+name+"/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# save model after each epoch
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1
)
path, dirs, files = next(os.walk('ML_Output_Files'))
file_count = len(files)
All_Trials=list(np.linspace(1,file_count,file_count).astype('int'))
Train_Trials=random.sample(All_Trials,int(file_count*.80))
All_Trials=[x for x in All_Trials if x not in Train_Trials]
Test_Trials=random.sample(All_Trials, int(file_count*.05))
Validation_Trials=[x for x in All_Trials if x not in Test_Trials]


X_samp=np.load('ML_Input_Files/UC_Design_1.npy'.format(1))
Y_samp=np.load('ML_Output_Files/UC_Design_C_1.npy'.format(1))
X_Train=np.zeros((len(Train_Trials),X_samp.shape[0],X_samp.shape[1]))
Y_Train=np.zeros((len(Train_Trials),Y_samp.shape[0]))

X_Test=np.zeros((len(Test_Trials),X_samp.shape[0],X_samp.shape[1]))
Y_Test=np.zeros((len(Test_Trials),Y_samp.shape[0]))
X_Val=np.zeros((len(Validation_Trials),X_samp.shape[0],X_samp.shape[1]))
Y_Val=np.zeros((len(Validation_Trials),Y_samp.shape[0]))
for i in range(0,len(Train_Trials)):
    X_Train[i,:,:]=np.load('ML_Input_Files/UC_Design_{}.npy'.format(Train_Trials[i]))
    Y_Train[i,:]=np.load('ML_Output_Files/UC_Design_C_{}.npy'.format(Train_Trials[i]))
for j in range(0,len(Test_Trials)):
    X_Test[j,:,:]=np.load('ML_Input_Files/UC_Design_{}.npy'.format(Test_Trials[j]))
    Y_Test[j,:]=np.load('ML_Output_Files/UC_Design_C_{}.npy'.format(Test_Trials[j]))
for k in range(0,len(Validation_Trials)):
    X_Val[k,:,:]=np.load('ML_Input_Files/UC_Design_{}.npy'.format(Validation_Trials[k]))
    Y_Val[k,:]=np.load('ML_Output_Files/UC_Design_C_{}.npy'.format(Validation_Trials[k]))    
X_Train=X_Train.reshape(X_Train.shape[0],X_Train.shape[1],X_Train.shape[2],1)
X_Test=X_Test.reshape(X_Test.shape[0],X_Test.shape[1],X_Test.shape[2],1)
X_Val=X_Val.reshape(X_Val.shape[0],X_Val.shape[1],X_Val.shape[2],1)
tensorboard_callback = TensorBoard(
    log_dir='tensorboard_logs/'+name,
    histogram_freq=1
)

Action='Test'

if Action=='Train': 
    history=model.fit(
        x=X_Train,
        y=Y_Train,
        epochs=40,
        verbose='auto',
        validation_data=(X_Val, Y_Val),
        batch_size=32,
        callbacks=[cp_callback]
    )
elif Action=='Test':
    model.load_weights('checkpoints/UC_Surrogate_Model_Weights_UpdatedRelu\cp.ckpt')
    print("Evaluate on test data")
    results = model.evaluate(X_Test, Y_Test, batch_size=10)
    print("test loss:", results[0])
