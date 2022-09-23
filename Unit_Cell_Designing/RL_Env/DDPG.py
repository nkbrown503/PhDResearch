# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:54:04 2022

@author: nbrow
"""


import os, time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
import matplotlib.pyplot as plt 

class Agent:
    def __init__(self, input_dims,Type, alpha, beta,gamma,Start_Noise,Noise_Decay,TN,
                 env=None,n_actions=7, max_size=20000, tau=0.005,
                 fc1=400, fc2=300, batch_size=256, noise=0.1):
        self.gamma = gamma
        self.tau = tau
        self.tau_decay=0
        self.tau_min=0.005
        self.TN=TN
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.Noise_steps=0
        self.Start_Noise=Start_Noise
        self.Noise_Decay=Noise_Decay
        self.Min_Noise=0.005
        self.Trial_Num=Type
        self.actor = ActorNetwork(n_actions=n_actions, name='actor_'+self.Trial_Num)
        self.critic = CriticNetwork(name='critic_'+self.Trial_Num)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         name='target_actor_'+self.Trial_Num)
        self.target_critic = CriticNetwork(name='target_critic_'+self.Trial_Num)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        self.tau=max([self.tau-self.tau_decay,self.tau_min])
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * self.tau + targets[i]*(1-self.tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * self.tau + targets[i]*(1-self.tau))

        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file).expect_partial()
        self.target_actor.load_weights(self.target_actor.checkpoint_file).expect_partial()
        self.critic.load_weights(self.critic.checkpoint_file).expect_partial()
        self.target_critic.load_weights(self.target_critic.checkpoint_file).expect_partial()

    def choose_action(self, observation, evaluate):
        self.Noise_steps+=1
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        self.noise=np.max([self.Start_Noise-(self.Noise_Decay*self.Noise_steps),self.Min_Noise])
      
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss,
                                                self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(
            critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        self.update_network_parameters()