# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:17:32 2021

@author: nbrow
"""


import argparse 

def parse_opts():
    parser=argparse.ArgumentParser()
    ''' Parameters invovled with generic environment building'''
    parser.add_argument('--Main_EX',
                        default=10,
                        type=int,
                        help='Number of X Elements for Larger Environment')
    
    parser.add_argument('--Main_EY',
                        default=10,
                        type=int,
                        help='Number of Y Elements for Larger Environment')
    
    parser.add_argument('--Sub_EX',
                    default=5,
                    type=int,
                    help='Number of X Elements for Smaller Environment used in Case of Progressive Refinement')
    
    parser.add_argument('--Sub_EY',
                    default=5,
                    type=int,
                    help='Number of Y Elements for Smaller Environment used in Case of Progressive Refinement')
    parser.add_argument('--Length_X',
                default=1,
                type=int,
                help='Length of the Structure in the X Direction')
    
    parser.add_argument('--Length_Y',
                default=1,
                type=int,
                help='Length of the Structure in the Y Direction')
    
    ''' Parameters Invovled with the TopOpt environment'''
    parser.add_argument('--Eta',
                    default=2,
                    type=int,
                    help='Used for dynamic adjusting reward function. Larger eta means lest prevelance given towards changes between current and previous reward. Recommend using [2,4]')
    parser.add_argument('--a',
                    default=5,
                    type=int,
                    help='X Coefficient of the Quadratic Reward Sufarce')
    
    parser.add_argument('--b',
                    default=5,
                    type=int,
                    help='Y Coefficient of the Quadratic Reward Sufarce')
    ''' Parameters Involved with the RL Architecture'''
    parser.add_argument('--Replace',
                    default=100,
                    type=int,
                    help='Number of iterations between switching the weights from the active network to the target network')
       
    parser.add_argument('--Epsilon_Decay',
                    default=5e-4,
                    type=float,
                    help='Iterative decay amount of the epsilon value used for exploration/explotation')
    parser.add_argument('--Epsilon_End',
                    default=0.01,
                    type=float,
                    help='Smallest Allowable Epsilon value to be used for exploration/explotation')
    
    parser.add_argument('--Memory_Size',
                    default=10000,
                    type=int,
                    help='Size of the Replay Buffer')
    parser.add_argument('--Num_Games',
                    default=50_000,
                    type=int,
                    help='Maximum Number of Training Episodes Conducted')
    
    parser.add_argument('--Batch_Size',
                    default=128,
                    type=int,
                    help='Batch Size that will be taken from the Replay Buffer per training episode')
    
    parser.add_argument('--LR',
                    default=5-3,
                    type=float,
                    help='Starting Learning Rate for the Network')
    
    parser.add_argument('--Gamma',
                    default=0.1,
                    type=float,
                    help='Discount Factor for Future Rewards ')
    args = parser.parse_args()

    return args
    
        