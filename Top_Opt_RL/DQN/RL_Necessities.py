# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:06:46 2021

@author: nbrow
"""
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras import layers,models 
from opts import parse_opts
opts=parse_opts()
class Agent():
    def __init__(self,env,Increase, lr, gamma, n_actions, epsilon, batch_size,
                 input_dims,filename_save,filename_load,EX,EY, epsilon_dec,mem_size, eps_end=opts.Epsilon_End, 
                 replace=opts.Replace):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions=n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.EX=EX
        self.EY=EY
        self.env=env
        self.eps_dec = epsilon_dec
        self.eps_min = eps_end
        self.replace = replace
        self.batch_size = batch_size
        self.lr=lr
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(self.EX*self.EY,Increase)
        self.q_next = DuelingDeepQNetwork(self.EX*self.EY,Increase)
        self.checkpoint_file_save='NN_Weights/'+filename_save+'_NN_weights'
        self.checkpoint_file_load='NN_Weights/'+filename_load+'_NN_weights'
        self.q_eval.compile(optimizer=Adam(learning_rate=self.lr),
                            loss='mean_squared_error')
        # just a formality, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=self.lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation,load_checkpoint,Testing):
        self.action_space = [i for i in range(self.n_actions)]
        if np.random.random() < self.epsilon and not load_checkpoint and not Testing:
            Void=np.array(self.env.VoidCheck)
            BC=np.array(np.reshape(self.env.BC_state,(1,(self.EX*self.EY))))
            LC=np.array(np.reshape(self.env.LC_state,(1,(self.EX*self.EY))))
            Clear_List=np.where(Void==0)[0]
            BC_List=np.where(BC==1)[0]
            LC_List=np.where(LC==1)[0]
            self.action_space = [ele for ele in self.action_space if ele not in Clear_List]
            self.action_space = [ele for ele in self.action_space if ele not in BC_List]
            self.action_space = [ele for ele in self.action_space if ele not in LC_List]
            action = np.random.choice(self.action_space)
        else:
            state = observation
            state=state.reshape(-1,self.EX,self.EY,3)
            actions = self.q_eval.call(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            Loss=.5
            return Loss

        if self.learn_step_counter % self.replace == 0 and self.learn_step_counter>0:
            self.q_next.set_weights(self.q_eval.get_weights())  

        states, actions, rewards, states_, dones = \
                                    self.memory.sample_buffer(self.batch_size)
        q_pred = self.q_eval(states)
        self.q_pred=q_pred
        q_next = self.q_next(states_)
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)
        # improve on my solution!
        for idx, terminal in enumerate(dones):
            #if terminal:
                #q_next[idx] = 0.0
            q_target[idx, actions[idx]] = rewards[idx] + \
                    self.gamma*q_next[idx, max_actions[idx]]*(1-int(dones[idx]))
        
        Loss=np.subtract(q_target,q_pred.numpy())
        Loss=np.square(Loss)
        Loss=Loss.mean()
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                        self.eps_min else self.eps_min 

        self.learn_step_counter += 1
        if self.learn_step_counter>5000:
            self.lr=2.5e-3
        if self.learn_step_counter>7500:
            self.lr=1e-3
        return Loss
        
    def save_models(self):
        print('... saving models ...')
        self.q_eval.save_weights(self.checkpoint_file_save)

    def load_models(self):
        print('... loading models ...')
        self.q_eval.load_weights(self.checkpoint_file_load)
        
class DuelingDeepQNetwork(keras.Model):
    def __init__(self, n_actions,Increase):
        super(DuelingDeepQNetwork, self).__init__()
        self.model = models.Sequential()
        if Increase:
            self.model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
    
        self.model.add(layers.Conv2D(16,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(8,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(4,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Conv2D(1,(3,3),padding='same',activation='relu'))
        self.model.add(layers.Flatten())
    
    def call(self, state):
        x = self.model(state)
    
        #V = self.model_V(x)
        #A = self.model_A(x)
        
        Q = x#V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))
        return Q
    
class ReplayBuffer():
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                        dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, new_states, dones
