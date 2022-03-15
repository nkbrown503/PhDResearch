# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:49:42 2022

@author: nbrow
"""
import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Dense, Input, Conv2D,AveragePooling2D, Flatten, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
class Actor_Model:
    def __init__(self, input_shape_UC,input_shape_Coef, action_space, lr, optimizer):
        UC_input = Input(shape=(20,60,1))
        Coef_input=Input(shape=(4,))
        self.action_space = action_space
        
        X_UC = Conv2D(64,(3,3), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(UC_input)
        
        X_UC = Conv2D(32,(3,3), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = AveragePooling2D(pool_size=(1,2))(X_UC)
        X_UC = Conv2D(16,(3,3), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = Conv2D(8 ,(3,3), activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = AveragePooling2D(pool_size=(1,2))(X_UC)
        X_UC = Conv2D(4 ,(3,3), activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = Conv2D(1 ,(3,3), activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)

        X_UC = AveragePooling2D(pool_size=(2,2))(X_UC)
        X_UC = Flatten()(X_UC)
        X_UC= Model(inputs=UC_input,outputs=X_UC)
        
        X_Coef= Dense(64,activation='relu')(Coef_input)
        X_Ceof= Dense(32,activation='relu')(X_Coef)
        X_Coef= Dense(16,activation='relu')(X_Coef)
        X_Coef= Dense(4,activation='relu')(X_Coef)
        X_Coef= Model(inputs=Coef_input,outputs=X_Coef)
        
        X_Comb= concatenate(([X_UC.output,X_Coef.output]))
        X_Comb= Dense(64,activation='relu')(X_Comb)
        X_Comb= Dense(32,activation='relu')(X_Comb)
        output = Dense(self.action_space, activation="sigmoid")(X_Comb)

        self.Actor = Model(inputs=(X_UC.input,X_Coef.input), outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(lr=lr))
        print(self.Actor.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state_UC,state_Coef):
        return self.Actor.predict([state_UC,state_Coef])


class Critic_Model:
    def __init__(self, input_shape_UC,input_shape_Coef, action_space, lr, optimizer):
        UC_input = Input(shape=(20,60,1))
        Coef_input=Input(shape=(4,))
        old_values = Input(shape=(1,))

        X_UC = Conv2D(64,(3,3), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(UC_input)
        
        X_UC = Conv2D(32,(3,3), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = AveragePooling2D(pool_size=(1,2))(X_UC)
        X_UC = Conv2D(16,(3,3), activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = Conv2D(8 ,(3,3), activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = AveragePooling2D(pool_size=(1,2))(X_UC)
        X_UC = Conv2D(4 ,(3,3), activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = Conv2D(1 ,(3,3), activation='relu', kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_UC)
        X_UC = AveragePooling2D(pool_size=(2,2))(X_UC)
        X_UC = Flatten()(X_UC)
        X_UC = Model(inputs=UC_input,outputs=X_UC)
        X_Coef= Dense(64,activation='relu')(Coef_input)
        X_Ceof= Dense(32,activation='relu')(X_Coef)
        X_Coef= Dense(16,activation='relu')(X_Coef)
        X_Coef= Dense(4,activation='relu')(X_Coef)
        X_Coef= Model(inputs=Coef_input,outputs=X_Coef)
        X_Comb= concatenate([X_UC.output,X_Coef.output])
        X_Comb= Dense(64,activation='relu')(X_Comb)
        X_Comb= Dense(32,activation='relu')(X_Comb)
        X_Comb= Dense(8,activation='relu')(X_Comb)
        value = Dense(1, activation=None)(X_Comb)

        self.Critic = Model(inputs=[X_UC.input,X_Coef.input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            #LOSS_CLIPPING = 0.2
            #clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            #v_loss1 = (y_true - clipped_value_loss) ** 2
            #v_loss2 = (y_true - y_pred) ** 2
            
            #value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state_UC,state_Coef):
        return self.Critic.predict([np.reshape(state_UC,(int(state_UC.shape[0]/20),20,60)),state_Coef, np.zeros((int(state_UC.shape[0]/20), 1))])