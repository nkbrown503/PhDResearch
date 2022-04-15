# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:45:14 2022

@author: nbrow
"""

from tensorflow.keras.layers import Input,Dense, Dropout, Multiply,Add, BatchNormalization, ReLU
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.models import Model
import keras.backend as K
import numpy as np
from tensorflow.keras.losses import Huber

def custom_loss(y_true,y_pred):
    loss=K.sum(K.mean(K.square(y_true-y_pred),axis=0))
    return loss
def FCC_model():

    inputs = Input(shape=(16,))
    t= Dense(256)(inputs)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(256)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(256)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(128)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(128)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(128)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(128)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    t=Dense(64)(t)
    t=ReLU()(t)
    t=BatchNormalization()(t)
    outputs = Dense(11)(t)
   

    lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=5e-3,
    decay_steps=20000,
    decay_rate=0.9)
    
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=custom_loss)


    return model