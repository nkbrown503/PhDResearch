# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 16:50:56 2022

@author: nbrow
"""
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta
from networks import Critic_Model, Actor_Model
import numpy as np
from memory import replay 
from tensorflow.keras import backend as K
import copy
import pylab
class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env):
        # Initialization
        # Environment and PPO parameters
        self.env = env
        self.action_size = self.env.action_space.shape[0]
        self.state_size_UC = [1,self.env.E_Y,self.env.E_X]
        self.state_size_Coef=self.env.Coef_Count
        self.EPISODES = 1000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = True
        self.env_name='UC_RLDesigner'
        self.Training_batch = 50
        #self.optimizer = RMSprop
        self.optimizer = Adam

        self.replay_count = 0
                
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots

        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape_UC=self.state_size_UC,input_shape_Coef=self.state_size_Coef, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Critic = Critic_Model(input_shape_UC=self.state_size_UC,input_shape_Coef=self.state_size_Coef, action_space = self.action_size, lr=self.lr, optimizer = self.optimizer)
        self.Actor_name = "weights/UC_Designer_PPO_Actor.h5"
        self.Critic_name = "weights/UC_Designer_PPO_Critic.h5"
        #self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        self.std = np.exp(self.log_std)


    def act(self, state_UC,state_Coef):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(np.reshape(state_UC,(1,20,60)),np.reshape(state_Coef,(1,4,)))

        low, high = -0.5, 0.5 # 0 and 1 are boundaries of sigmoid
        action = pred + np.random.uniform(low, high, size=pred.shape) * self.std
        action = np.clip(action, 0, 1)
        
        logp_t = self.gaussian_likelihood(action, pred, self.log_std)

        return action, logp_t

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)

    def discount_rewards(self, reward):#gaes is better
        # Compute the gamma-discounted rewards over an episode
        # We apply the discount and normalize it to avoid big variability of rewards
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= (np.std(discounted_r) + 1e-8) # divide by standard deviation
        return discounted_r
    def replay(self,states_UC,states_Coef, actions, rewards, dones, next_states_UC,next_states_Coef, logp_ts):
            # reshape memory to appropriate shape for training
            states_UC = np.vstack(states_UC)
            states_Coef= np.vstack(states_Coef)
            next_states_UC = np.vstack(next_states_UC)
            next_states_Coef=np.vstack(next_states_Coef)
            actions = np.vstack(actions)
            logp_ts = np.vstack(logp_ts)
    
            # Get Critic network predictions 
            values = self.Critic.predict(states_UC,states_Coef)

            next_values = self.Critic.predict(next_states_UC,next_states_Coef)

            # Compute discounted rewards and advantages
            #discounted_r = self.discount_rewards(rewards)
            #advantages = np.vstack(discounted_r - values)
            advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
            '''
            pylab.plot(adv,'.')
            pylab.plot(target,'-')
            ax=pylab.gca()
            ax.grid(True)
            pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
            pylab.show()
            if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
            '''

            # stack everything to numpy array
            # pack all advantages, predictions and actions to y_true and when they are received
            # in custom loss function we unpack it
            y_true = np.hstack([advantages, actions, logp_ts])
            
            # training Actor and Critic networks
            a_loss = self.Actor.Actor.fit([np.reshape(states_UC,(int(states_UC.shape[0]/20),20,60)),states_Coef], y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
            c_loss = self.Critic.Critic.fit([np.reshape(states_UC,(int(states_UC.shape[0]/20),20,60)),states_Coef,values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
    
            # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
            pred = self.Actor.predict(np.reshape(states_UC,(int(states_UC.shape[0]/20),20,60)),states_Coef)
            log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
            logp = self.gaussian_likelihood(actions, pred, log_std)
            approx_kl = np.mean(logp_ts - logp)
            approx_ent = np.mean(-logp)
    
            self.replay_count += 1
    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.90, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)


    def load(self):
        self.Actor.Actor.load_weights(self.Actor_name)
        self.Critic.Critic.load_weights(self.Critic_name)

    def save(self):
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)


    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and save:
            self.max_average = self.average_[-1]
            self.save()
            SAVING = "SAVING"
            # decreaate learning rate every saved model
            #self.lr *= 0.99
            #K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
            #K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run_batch(self):
        state_UC,state_Coef = self.env.reset()
        reward_tracker=[]
        done,score=False, 0

        while True:
            # Instantiate or reset games memory
            states_UC,states_Coef, next_states_UC,next_states_Coef, actions, rewards, dones, logp_ts = [], [], [], [], [], [], [], []
            for t in range(self.Training_batch):
                # Actor picks an action
                action, logp_t = self.act(state_UC,state_Coef)
                if self.episode%20==0:
                    print(logp_t)
                # Retrieve new state, reward, and whether the state is terminal
                next_state_UC,next_state_Coef, reward, done, Legal = self.env.step(action[0])

                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states_UC.append(state_UC)
                states_Coef.append(state_Coef)
                next_states_UC.append(next_state_UC)
                next_states_Coef.append(next_state_Coef)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state_UC = next_state_UC
                state_Coef= next_state_Coef
                score += reward

                #self.env.render(Legal)
                if done:
                    self.episode += 1
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))

                    
                    state_UC,state_Coef= self.env.reset()
                    done, score=False,0

            self.replay(states_UC,states_Coef, actions, rewards, dones, next_states_UC,next_states_Coef,logp_ts)
            if self.episode >= self.EPISODES:
                break

       


    def run_multiprocesses(self, num_worker = 4):
        works, parent_conns, child_conns = [], [], []
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True)
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states =        [[] for _ in range(num_worker)]
        next_states =   [[] for _ in range(num_worker)]
        actions =       [[] for _ in range(num_worker)]
        rewards =       [[] for _ in range(num_worker)]
        dones =         [[] for _ in range(num_worker)]
        logp_ts =       [[] for _ in range(num_worker)]
        score =         [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            # get batch of action's and log_pi's
            action, logp_pi = self.act(np.reshape(state, [num_worker, self.state_size[0]]))
            
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])

            for worker_id, parent_conn in enumerate(parent_conns):
                next_state, reward, done, _ = parent_conn.recv()

                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                score[worker_id] += reward

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                        
                        
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    self.replay(states[worker_id], actions[worker_id], rewards[worker_id], dones[worker_id], next_states[worker_id], logp_ts[worker_id])

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []

        # terminating processes after a while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 1):#evaluate
        self.load()
        for e in range(test_episodes):
            state_UC,state_Coef = self.env.reset()
            done = False
            score = 0
            Legal=False
            while not done:
                self.env.render(Legal)
                action = self.Actor.predict(np.reshape(state_UC,(1,20,60)),np.reshape(state_Coef,(1,4,)))[0]
                print(action)
                next_state_UC,next_state_Coef, reward, done, Legal = self.env.step(action)
                state_UC = next_state_UC
                state_Coef= next_state_Coef
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()