# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 09:29:09 2021

@author: nbrow
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import math
def plot_learning_curve(x, scores):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-250):(i+1)])
        plt.plot(x, running_avg)
        plt.title('CNN Layer Effect on Maximum Reward')
        plt.xlabel('Episodes')
        plt.ylabel(' Average Reward')
        
        #plt.savefig(figure_file)
DN1='Trial_Data/DDQN_TopOpt_Generalized_CNN_4L_Gen_10by10_FailTestData.pkl'


Data1_pd=pd.read_pickle(DN1)
Data1_pd=Data1_pd.drop_duplicates()
Data1_pd = Data1_pd.drop(Data1_pd[Data1_pd.Score5 > 40].index)

#Data1_pd.to_pickle('FailedBC_LC.pkl')
Data1_pd=Data1_pd.to_numpy()
Mean=np.mean(Data1_pd[:,10])
Mean2=np.mean(Data1_pd[:,11])
print(Mean)
print(Mean2)


#x1=range(0,6200)

#plot_learning_curve(x1,Data1_pd[0:6200,6])

#plt.legend(['Loss'])
