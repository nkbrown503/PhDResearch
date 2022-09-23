# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:18:09 2022

@author: nbrow
"""


import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, LeakyReLU, ReLU

class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256,
            name='critic_'+'2', chkpt_dir='weights_48'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg')
        self.fc0 = Dense(5096,activation='relu')
        self.fc1 = Dense(2048,activation='relu')
        self.fc2 = Dense(1024,activation='relu')
        self.fc3 = Dense(512,activation='relu')
        self.fc4 = Dense(256,activation='relu')
        #self.fc4 = Dense(32)

        self.q = Dense(1, activation=None)

    def call(self, state, action):
        
        action_value = self.fc0(tf.concat([state, action], axis=1))
        action_value = self.fc1(action_value)
        action_value = self.fc2(action_value)

        action_value = self.fc3(action_value)

        action_value = self.fc4(action_value)


        q = self.q(action_value)

        return q

class ActorNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, n_actions=7, name='actor_'+'2',
            chkpt_dir='weights_48'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                    self.model_name+'_ddpg')
        self.fc0 = Dense(5096,activation='relu')
        self.fc1 = Dense(2048,activation='relu')
        self.fc2 = Dense(1024,activation='relu')
        self.fc3 = Dense(512,activation='relu')
        self.fc4 = Dense(256,activation='relu')

        self.mu = Dense(self.n_actions, activation='sigmoid')

    def call(self, state):
        prob = self.fc0(state)
        prob = self.fc1(prob)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        prob = self.fc4(prob)

        mu = self.mu(prob)

        return mu
