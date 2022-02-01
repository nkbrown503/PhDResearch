# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:50:16 2022

@author: nbrow
"""


import numpy as np 
import matplotlib.pyplot as plt 
import sys 
from scipy.optimize import curve_fit
Tess=[]
Iterations=5000
X_Fit_C=list(np.arange(-0.2,0+0.2/10,0.2/10))
X_Fit_T=list(np.arange(0,0.25+0.25/10,0.25/10))
avg_error=[]
def func(x,a,b,c):
    return (a*x**3)+(b*x**2)+(c*x)
        
Bound=20
for It in range(0,Iterations):

    FileName_C='UC_Design_AR3_C_Trial{}'.format(int(It+1))
    #FileName_T='UC_Design_AR3_T_Trial{}'.format(int(It+1))

    Results_C=np.flip(np.load('Result_Files/'+FileName_C+'.npy'),axis=0)

    #Results_T=abs(np.load('Result_Files/'+FileName_T+'.npy'))
    
    C_Fit=np.polyfit(Results_C[:,0]/(120*3),Results_C[:,1]/(1e4*360),3)
    popt,pcov=curve_fit(func,Results_C[:,0]/(120*3),Results_C[:,1]/(1e4*360),bounds=(0,Bound))

    #T_Fit=np.polyfit(Results_T[:,0]/(120*3),Results_T[:,1]/(1e4*360),3)
    np.save('ML_Output_Files/UC_Design_C_{}.npy'.format(It+1),popt)
    #np.save('ML_Output_Files/UC_Design_T_{}.npy'.format(It+1),T_Fit[0:3])

    Y_Fit_C=[(Strain**3*popt[0])+(Strain**2*popt[1])+(Strain*popt[2]) for Strain in X_Fit_C] #+(Strain**2*C_Fit[2])+(Strain*C_Fit[3])
    Y_Fit_C[-1]=0
    #change_percent = abs(((Y_Fit_C[0:10]-(Results_C[0:10,1]/(1e4*360)))/(Results_C[0:10,1]/(1e4*360)))*100)
    #avg_error=np.append(avg_error,np.mean(change_percent))

    
    #Y_Fit_T=[(Strain**3*T_Fit[0])+(Strain**2*T_Fit[1])+(Strain*T_Fit[2]) for Strain in X_Fit_T]
    
    if It%1000==0:
        plt.plot(Results_C[:,0]/(120*3),Results_C[:,1]/(1e4*360),'-',label='FEA Results C')
        plt.plot(X_Fit_C,Y_Fit_C,'--',label='Curve Fit Results C')
        #plt.plot(Results_T[:,0]/(120*3),Results_T[:,1]/(1e4*360),'-',label='FEA Results T')
        #plt.plot(X_Fit_T,Y_Fit_T,'-',label='Curve Fit Results T')
        plt.legend(loc='best')
        plt.xlabel('Strain')
        plt.ylabel('Stress [10E4]')
        plt.axis([-.2,.2,-0.4,0])
        
#print('The average Error for a Bound Value of [-{}, {}] is {}'.format(Bound,Bound,np.mean(avg_error)))

    
    