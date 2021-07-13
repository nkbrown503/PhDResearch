# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 09:34:14 2021

@author: nbrow
"""


''' Nathan Brown 
Policy Gradient Training of Topology Optimization through Reinforcement learning'''
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers,datasets,models 
from tensorflow.keras import Sequential
from Mesh_Transformer import Mesh_Transform
import pandas as pd
import matplotlib.pyplot as plt
import time
import statistics 
import random
from statistics import stdev
from gym import Env
from gym.spaces import Discrete, Box
import FEA_SOLVER_GENERAL
import itertools
import copy 
import keras.backend as K
#Define The Size and Scope of Your Training'
# Before the end check if any more material could be removed. If it could, then you penalize, if not then you give additional reward for achieving a final topology 
Data=pd.read_pickle('Trial_Data/Reward_Data.pkl')
Data=Data.to_numpy()
X_Data=Data[:,0]
Y_Data=Data[:,1]
Z_Data=Data[:,2]
def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G
ordr=2
GG = poly_matrix(X_Data, Y_Data, ordr)
# Solve for np.dot(G, m) = z:
mm = np.linalg.lstsq(GG, Z_Data)[0]
nx, ny = 1000, 1000

xx, yy = np.meshgrid(np.linspace(0, 1, nx),
                     np.linspace(0, 1, ny))
GoGG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
Reward_Ind = np.reshape(np.dot(GoGG, mm), xx.shape)[:,-1]

Mesh_X=[5]
Mesh_Y=[5]

Num_hidden=[128,128,128]
VoidCheck=np.ones((1,Mesh_X[0]*Mesh_Y[0]))
VoidCheck=list(VoidCheck[0])
Vol_Fraction=[10/16,22/64,19/49]
Ep_decay=[5e-4,5e-5,7.5e-5]
Mem_Size=[5000,25000,10000]
for k in range(len(Mesh_X)):
    Lx=1
    Ly=1
    Elements_X=Mesh_X[k]
    Elements_Y=Mesh_Y[k]
    ElementSize=Elements_X*Elements_Y
    NodeSize=(Elements_X+1)*(Elements_Y+1)
    ElementNodes=FEA_SOLVER_GENERAL.rectangularmesh(Lx,Ly,Elements_X,Elements_Y)[1]
    #Vol_fraction=Vol_Fraction[k]
    #Remove_num=ElementSize-(ElementSize*Vol_fraction)

    #---Loading Boundary Conditions----
    a=1.2
    b=10
    
    tic=time.perf_counter()
    x=np.array([1,0,0,1,.5,0,.5])
    y=np.array([0,0,1,1,.5,.5,0])
    z=np.array([])
    Go_List=[]
    Go_List=np.append(Go_List,range(0,Elements_X+1))
    for num in range(0,Elements_Y-1):
        Go_List=np.append(Go_List,(Elements_X+1)*(num+1))
        Go_List=np.append(Go_List,(Elements_X+1)*(num+2)-1)
    Go_List=np.append(Go_List,range((Elements_X*(Elements_Y+1)),(Elements_X*(Elements_Y+2)+1)))
          
    for i in range(0,len(x)):
        z=np.append(z,(a*(x[i])**2)+(b*(y[i])**2))
    
    ordr=2
    G = poly_matrix(x, y, ordr)
    # Solve for np.dot(G, m) = z:
    m = np.linalg.lstsq(G, z)[0]
    nx, ny = 1000, 1000

    xx, yy = np.meshgrid(np.linspace(0, 1, nx),
                         np.linspace(0, 1, ny))
    GoG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
    zz = np.reshape(np.dot(GoG, m), xx.shape)
    #LeftBC=list(range(0,Elements_X*(Elements_Y-1)*2,Elements_X*(Elements_Y-1)))
    #LeftBC.append(Elements_X-1) #For Bottom Right
    filename = 'DDQN_TopOpt_Generalized_R3_'+str(Mesh_X[k])+'by'+str(Mesh_Y[k])
    class TopOpt_Gen(Env):
        def __init__(self):
            #Actons we can take... remove any of the blocks
            self.action_space=Discrete(ElementSize)
            #Current Material Matrix
            Low=np.array([0]*ElementSize)
            High=np.array([1]*ElementSize)
            self.observation_space=Box(Low,High,dtype=np.float32)

            # Set Maximum Number of Blocks to Remove
            #self.Remove_Tot=Remove_num
            
        def step(self,action,observation):
            #Apply Action
    
                
            # evaluate it on grid
            rs_place=self.VoidCheck[action]
            self.VoidCheck[action]=0
            ElementMat=np.reshape(self.VoidCheck,(Elements_X,Elements_Y))
            SingleCheck=FEA_SOLVER_GENERAL.isolate_largest_group_original(ElementMat)
            It=list(self.VoidCheck).count(0)
            if rs_place==1 and action not in self.BC and SingleCheck[1]==True:
                done=False
                   
                Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),Lx,Ly,Elements_X,Elements_Y,self.Loaded_Node,self.Load_Direction,self.BC1,self.BC2,Stress=True)
        
                Max_SE_Ep=np.max(Run_Results[1])

                reward = zz[(int((self.Max_SE_Tot/Max_SE_Ep)*1000))-1,int((It/ElementSize)*1000)-1]
                reward2=Reward_Ind[int(1-(np.reshape(self.Stress_state,(Elements_X*Elements_Y,1))[action])*1000)-1]
                reward=statistics.mean([reward,reward2])
                self.Stress_state=Run_Results[3]
                self.Stress_state=np.reshape(self.Stress_state,(Elements_X,Elements_Y))
                self.state=np.zeros((Elements_X,Elements_Y,3))
                self.state[:,:,0]=self.Stress_state
                self.state[:,:,1]=self.BC_state
                self.state[:,:,2]=self.LC_state
                    
            else:
                """If the removed block has already been removed, leads to a non-singular
                body or one of the Boundary condition blocks, the agent should be severely punished (-10)"""
                Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),Lx,Ly,Elements_X,Elements_Y,self.Loaded_Node,self.Load_Direction,self.BC1,self.BC2,Stress=True)
                Max_SE_Ep=np.max(Run_Results[1])
                self.Stress_state=Run_Results[3]
                self.Stress_state=np.reshape(self.Stress_state,(Elements_X,Elements_Y))
                self.state=np.zeros((Elements_X,Elements_Y,3))
                self.state[:,:,0]=self.Stress_state
                self.state[:,:,1]=self.BC_state
                self.state[:,:,2]=self.LC_state
                reward=-1
                done=True
                Max_SE_Ep=0
                if rs_place==1:
                    self.VoidCheck[action]=1
                Solid=[i for i, e in enumerate(self.VoidCheck) if e == 1]
                Not_Finish=False
                Check_Void=copy.deepcopy(self.VoidCheck)

                for mn in Solid:
                    Check_Void[mn]=0
                    Check_ElementMat=np.reshape(Check_Void,(Elements_X,Elements_Y))
                    SingleCheck=FEA_SOLVER_GENERAL.isolate_largest_group_original(Check_ElementMat)
                    Check_Void[mn]=1
                    if SingleCheck[1]==True:
                        Not_Finish=True
                        break
                if Not_Finish==False:
                    Run_Results=FEA_SOLVER_GENERAL.FEASolve(list(self.VoidCheck),Lx,Ly,Elements_X,Elements_Y,self.Loaded_Node,self.Load_Direction,self.BC1,self.BC2,Stress=True)
                    Max_SE_Ep=np.max(Run_Results[1])
                    reward = zz[(int((self.Max_SE_Tot/Max_SE_Ep)*1000))-1,int((It/ElementSize)*1000)-1]
                    reward2=Reward_Ind[int(1-(np.reshape(self.Stress_state,(Elements_X*Elements_Y,1))[action])*1000)-1]
                    reward=statistics.mean([reward,reward2])

                    
                    
                
            #PLaceholder for Info
            info={}
    
            return self.state, reward, done, It,info
        
        def render(self, mode='human'):
            RenderMat=copy.deepcopy(self.VoidCheck)
            RenderMat[self.BC1_Element]=2
            RenderMat[self.BC2_Element]=2
            RenderMat[self.Loaded_Element]=4
            RenderMat=np.reshape(RenderMat,(Elements_X,Elements_Y))
            print(np.flip(RenderMat,0))
            print('')
            return 
            
        def reset(self):

            self.VoidCheck = VoidCheck
            self.VoidCheck=np.array(self.VoidCheck)
            self.Stress_state =FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,self.Loaded_Node,self.Load_Direction,self.BC1,self.BC2,Stress=True)[3]
            #self.Stress_state=list(np.array(self.Stress_state)
            self.Stress_state=np.reshape(self.Stress_state,(Elements_X,Elements_Y))
            self.state=np.zeros((Elements_X,Elements_Y,3))
            self.state[:,:,0]=self.Stress_state
            self.state[:,:,1]=self.BC_state
            self.state[:,:,2]=self.LC_state
            
            return self.state
        def reset_conditions(self):
            self.Max_SE_Tot=0
            while self.Max_SE_Tot<=0 or self.Max_SE_Tot>5000:
                self.BC1=random.choice([i for i in Go_List])
                self.BC2=random.choice([i for i in Go_List])
                self.BC1_Element=np.where(ElementNodes==self.BC1)[0][0]
                self.BC2_Element=np.where(ElementNodes==self.BC2)[0][0]
                BC_set=[self.BC1_Element,self.BC2_Element]
                self.Loaded_Node=int(random.choice([i for i in Go_List]))
                self.Loaded_Element=np.where(ElementNodes==self.Loaded_Node)[0][0]
                while self.Loaded_Element in BC_set:
                    self.Loaded_Node=int(random.choice([i for i in Go_List]))
                    self.Loaded_Element=np.where(ElementNodes==self.Loaded_Node)[0][0]
                
                self.LC_state=list(np.zeros((1,Elements_X*Elements_Y))[0])
                self.LC_state[self.Loaded_Element]=1
                self.LC_state=np.reshape(self.LC_state,(Elements_X,Elements_Y))
                self.Load_Type=random.choice([0,1])
                if self.Load_Type==0: #Load will be applied vertically
                    self.Loaded_Node=self.Loaded_Node+NodeSize
                    
                self.Load_Direction=random.choice([-1,1]) #1 for Compressive Load, -1 for tensile load
                self.BC=[self.BC1_Element,self.BC2_Element,self.Loaded_Element]
                self.BC_state=list(np.zeros((1,Elements_X*Elements_Y))[0])
                self.BC_state[self.BC1_Element]=1
                self.BC_state[self.BC2_Element]=1
                self.BC_state=np.reshape(self.BC_state,(Elements_X,Elements_Y))
                self.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,self.Loaded_Node,self.Load_Direction,self.BC1,self.BC2,Stress=True)[1]))

    env = TopOpt_Gen()
    
    class DuelingDeepQNetwork(keras.Model):
        def __init__(self, n_actions):
            super(DuelingDeepQNetwork, self).__init__()
            self.model = models.Sequential()
            self.model.add(layers.Conv2D(1,(1,1),strides=(1,1),activation='relu',input_shape=(Elements_X,Elements_Y,3)))
            #self.model.add(layers.MaxPooling2D((1,1)))
            #self.model.add(layers.Conv2D(1,(1,1),activation='relu'))
            self.model.add(layers.Flatten())
            #self.model.add(layers.Dense(45,activation='relu'))
            #self.model.add(layers.Dense(32,activation='relu'))
            #self.model.add(layers.Dense(32,activation='relu'))
            #self.model.add(layers.Dense(25,activation='relu'))
            #self.model.add(layers.Dense(25,activation='relu'))
         
            
            
            self.V = layers.Dense(1, activation=None)
            self.A = layers.Dense(n_actions, activation=None)
    
        def call(self, state):
            x = self.model(state)
                        
            V = self.V(x)
            A = self.A(x)
    
            Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
    
            return Q
    
        def advantage(self, state):
            x = self.model(state)

            A = self.A(x)
    
            return A
    
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
    
    class Agent():
        def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                     input_dims,filename, epsilon_dec=Ep_decay[k], eps_end=0.01, 
                     mem_size=Mem_Size[k], fc1_dims=Num_hidden[k],
                     fc2_dims=Num_hidden[k], replace=50):
            self.action_space = [i for i in range(n_actions)]
            self.gamma = gamma
            self.epsilon = epsilon
            self.eps_dec = epsilon_dec
            self.eps_min = eps_end
            self.replace = replace
            self.batch_size = batch_size
            self.lr=lr
            self.learn_step_counter = 0
            self.memory = ReplayBuffer(mem_size, input_dims)
            self.q_eval = DuelingDeepQNetwork(Elements_X*Elements_Y)
            self.q_next = DuelingDeepQNetwork(Elements_X*Elements_Y)
            self.checkpoint_file='NN_Weights/'+filename+'_NN_weights'
            self.q_eval.compile(optimizer=Adam(learning_rate=self.lr),
                                loss='mean_squared_error')
            # just a formality, won't optimize network
            self.q_next.compile(optimizer=Adam(learning_rate=self.lr),
                                loss='mean_squared_error')
    
        def store_transition(self, state, action, reward, new_state, done):
            self.memory.store_transition(state, action, reward, new_state, done)
    
        def choose_action(self, observation,load_checkpoint,Testing):
            if np.random.random() < self.epsilon and load_checkpoint is False and Testing is False:
                action = np.random.choice(self.action_space)
            else:

                state = observation
                
                state=state.reshape(-1,Elements_X,Elements_Y,3)
                actions = self.q_eval.call(state)
        
                action = tf.math.argmax(actions, axis=1).numpy()[0]
            return action
    
        def learn(self):

            if self.memory.mem_cntr < self.batch_size:
                Loss=.5
                return Loss
    
            if self.learn_step_counter % self.replace == 0:
                agent.q_next.set_weights(agent.q_eval.get_weights())  
    
            states, actions, rewards, states_, dones = \
                                        self.memory.sample_buffer(self.batch_size)
    
            q_pred = self.q_eval(states)
    
            self.q_pred=q_pred
            q_next = self.q_next(states_)
            # changing q_pred doesn't matter because we are passing states to the train function anyway
            # also, no obvious way to copy tensors in tf2?
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
            if self.learn_step_counter>2000:
                self.lr=1e-3
            if self.learn_step_counter>4000:
                self.lr=5e-4
            return Loss
            
        def save_models(self):
            print('... saving models ...')
            self.q_eval.save_weights(self.checkpoint_file)
    
        def load_models(self):
            print('... loading models ...')
            self.q_eval.load_weights(self.checkpoint_file)
            
    def plot_learning_curve(x, scores, figure_file):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.xlabel('Episodes')
        plt.ylabel(' Average Reward')
        plt.savefig(figure_file)
        
            
    
    agent = Agent(lr=5e-3, gamma=0,filename=filename, n_actions=env.action_space.n, epsilon=1.0,
                      batch_size=64, input_dims=[Elements_X,Elements_Y,3])
    n_games = 1_000_000
    # uncomment this line and do a mkdir tmp && mkdir video if you want to
    # record video of the agent playing the game.
    
    
    figure_file = 'plots/' + filename +'_reward.png'    
    best_score = env.reward_range[0]
    best_test_score=0
    total_score=0
    score_history = []
    step_history=[]
    per_history=[]
    succ_history=[]
    Loss_history=[]
    Ep_History=[]
    
    load_checkpoint=False
    Retrain=False
    if load_checkpoint:
        agent.load_models()
        n_games=10
    if not load_checkpoint:
        TrialData=pd.DataFrame(columns=['Episode','Reward','Successfull Steps','Percent Successful','Avg Loss','SDEV','Epsilon','Time'])
    env.reset_conditions()
    for i in range(n_games):
        Testing = False
        if load_checkpoint:
            env.BC1=int(input('Please select a node to apply Boundary condition #1: '))
            env.BC2=int(input('Please select a node to apply Boundary condition #2: '))
            env.Loaded_Node=int(input('Please select a node to apply the load to: '))
            env.Load_Type=int(input('Input 0 for a Vertical Load or Input 1 for a Horizontal load: '))
            env.Load_Direction=int(input('Input -1 for a tensile load or Input 1 for a compressive load: '))
            env.Loaded_Element=np.where(ElementNodes==env.Loaded_Node)[0][0]
            if env.Load_Type==0:
                env.Loaded_Node+=NodeSize
            env.BC1_Element=np.where(ElementNodes==env.BC1)[0][0]
            env.BC2_Element=np.where(ElementNodes==env.BC2)[0][0]
            env.LC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
            env.LC_state[env.Loaded_Element]=1
            env.LC_state=np.reshape(env.LC_state,(Elements_X,Elements_Y))
            env.BC=[env.BC1_Element,env.BC2_Element,env.Loaded_Element]
            env.BC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
            env.BC_state[env.BC1_Element]=1
            env.BC_state[env.BC2_Element]=1
            env.BC_state=np.reshape(env.BC_state,(Elements_X,Elements_Y))
            env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Load_Direction,env.BC1,env.BC2,Stress=True)[1]))

        done = False
        score = 0
        if i%10==0 and i>=200:
            Testing=True
            if i%100==0:
                env.BC1=0
                env.BC2=Elements_X*(Elements_Y+1)
                env.Loaded_Node=5+(NodeSize)
                env.Loaded_Element=np.where(ElementNodes==env.Loaded_Node-NodeSize)[0][0]
                env.BC1_Element=np.where(ElementNodes==env.BC1)[0][0]
                env.BC2_Element=np.where(ElementNodes==env.BC2)[0][0]
                env.LC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
                env.LC_state[env.Loaded_Element]=1
                env.LC_state=np.reshape(env.LC_state,(Elements_X,Elements_Y))
    
                env.Load_Direction=-1 #1 for Compressive Load, -1 for tensile load
                env.BC=[env.BC1_Element,env.BC2_Element,env.Loaded_Element]
                env.BC_state=list(np.zeros((1,(Elements_X)*(Elements_Y)))[0])
                env.BC_state[env.BC1_Element]=1
                env.BC_state[env.BC2_Element]=1
                env.BC_state=np.reshape(env.BC_state,(Elements_X,Elements_Y))
                env.Max_SE_Tot=np.max((FEA_SOLVER_GENERAL.FEASolve(VoidCheck,Lx,Ly,Elements_X,Elements_Y,env.Loaded_Node,env.Load_Direction,env.BC1,env.BC2,Stress=True)[1]))

            print('--------Testing Run------')
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation,load_checkpoint,Testing)
            tic2=time.perf_counter()
            observation_, reward, done, It,info = env.step(action,observation)
            agent.store_transition(observation,action,reward,observation_,done)
            score += reward
            
            if Testing is True:
                env.render()
                print('Current Score: '+str(round(score,3)))
            observation = observation_
            if load_checkpoint:
                env.render()
        if load_checkpoint:
            if env.Load_Type==0:
                Load_Type='Vertical'
            else:
                Load_Type='Horizontal'
            if env.Load_Direction==-1:
                Load_Dir='Tensile'
            else:
                Load_Dir='Compressive'
            print('----------------')
            print('The final topology with BCs located at nodes '+str(env.BC1)+' and '+str(env.BC2)+' with a '+Load_Type+' '+Load_Dir+' load applied to node '+str(env.Loaded_Node)+': \n')
            env.render()
            print('----------------')
                  
        if not load_checkpoint:
            Total_Loss=agent.learn()
        else:
            Total_Loss=1

        Loss_history.append(Total_Loss)
        avg_Loss=np.mean(Loss_history[-50:])
        
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        if i>1:
            score_std = stdev(score_history[-50:])
        else:
            score_std=1
        if Testing:
            print('Current Best Testing Score: '+str(round(best_test_score,3)))
        if not load_checkpoint:
            env.reset_conditions()
        if avg_score>=best_score:
            agent.save_models()
            best_score=avg_score
        if Testing is True and not load_checkpoint and score>best_test_score and i>1:
            best_test_score=score
            print('New Best Testing Reward')
            #agent.q_next.set_weights(agent.q_eval.get_weights())
            agent.save_models()
        if load_checkpoint:
            avg_Loss=1
    
        #if avg_score >= best_score and not load_checkpoint:
        #    best_score = avg_score
        #    env.render()
        #    Best_VoidCheck=env.VoidCheck
        #    #agent.q_next.set_weights(agent.q_eval.get_weights())
            
            
        #    if not load_checkpoint:
        #        agent.save_models()
    
        Succ_Steps=list(env.VoidCheck).count(0)
        succ_history.append(Succ_Steps)
    
        avg_succ = np.mean(succ_history[-50:])
        Percent_Succ=Succ_Steps/ElementSize
        per_history.append(Percent_Succ)
        avg_percent=np.mean(per_history[-50:])
        toc=time.perf_counter()
        if not load_checkpoint:
            TrialData=TrialData.append({'Episode': i, 'Reward': score,'Successfull Steps': Succ_Steps,
                    'Percent Successful':Percent_Succ,'Avg Loss':avg_Loss,'SDev.':score_std,'Epsilon': agent.epsilon, 'Time':round((toc-tic),3)}, ignore_index=True)
    
        print('Episode ', i, '  Score %.2f' % score,'  Avg_score %.2f' % avg_score,'  Avg Steps %.0f' % avg_succ,'   Avg Percent %.0f' %(avg_percent*100),' SDev %.2f' %score_std,'     Avg Loss %.2f' %avg_Loss,'  Ep.  %.2f' %agent.epsilon,'  Time (s) %.0f' %(toc-tic))
        if i%100==0 and not load_checkpoint:
            TrialData.to_pickle('Trial_Data/'+filename +'_TrialData.pkl')
        if not load_checkpoint and i%100==0 and i>0 :
            x = range(0,i+1)
            plot_learning_curve(x, score_history, figure_file)
     
        if not load_checkpoint and i==n_games-1 :
            x = range(0,i+1)
            plot_learning_curve(x, score_history, figure_file)
      
        if avg_percent>.95 and score_std<=abs(0.01*avg_score) and not load_checkpoint:
    
            print('Solved in '+str(i)+'iterations.')
            x = range(0,i+1)
            plot_learning_curve(x, score_history, figure_file)
    
            TrialData.to_pickle('Trial_Data/'+filename+'_TrialData.pkl')
            break
    #VoidCheck=Best_VoidCheck
    if not load_checkpoint:
        print('------------------------------------')
        print('Currently Proposed Optimal Topology')
        print(np.flip(np.reshape(VoidCheck,(Mesh_Y[k],Mesh_X[k])),0))
        print('------------------------------------')
        if k<len(Mesh_X)-1:
            VoidCheck=Mesh_Transform(Mesh_X[k],Mesh_Y[k],Mesh_X[k+1],Mesh_Y[k+1],VoidCheck)
            print('------------------------------------')
            print('Corresponding Topology for Mesh Refinement')
            print(np.flip(np.reshape(VoidCheck,(Mesh_Y[k+1],Mesh_X[k+1])),0))
            print('------------------------------------')
    
    
    
