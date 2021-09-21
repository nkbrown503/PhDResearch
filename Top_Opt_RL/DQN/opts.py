# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:17:32 2021

@author: nbrow
"""


import argparse 

def parse_opts(args_in = None):
    parser=argparse.ArgumentParser()
    ''' Parameters invovled with generic environment building'''
    parser.add_argument('--Main_EX',
                        default=24,
                        type=int,
                        help='Number of X Elements for Larger Environment')
    
    parser.add_argument('--Main_EY',
                        default=24,
                        type=int,
                        help='Number of Y Elements for Larger Environment')
    parser.add_argument('--PR2_EX',
                    default=12,
                    type=int,
                    help='Number of X Elements for Second Environment used in Case of Progressive Refinement')
    parser.add_argument('--PR2_EY',
                    default=12,
                    type=int,
                    help='Number of Y Elements for Second Environment used in Case of Progressive Refinement')
    parser.add_argument('--PR_EX',
                    default=6,
                    type=int,
                    help='Number of X Elements for Smaller Environment used in Case of Progressive Refinement')
    
    parser.add_argument('--PR_EY',
                    default=6,
                    type=int,
                    help='Number of Y Elements for Smaller Environment used in Case of Progressive Refinement')
    
    parser.add_argument('--Lx',
                default=1,
                type=int,
                help='Length of the Structure in the X Direction')
    
    parser.add_argument('--Ly',
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
    parser.add_argument('--replace',
                    default=100,
                    type=int,
                    help='Number of iterations between switching the weights from the active network to the target network')
       
    parser.add_argument('--epsilon_dec',
                    default=3.5e-4,
                    type=float,
                    help='Iterative decay amount of the epsilon value used for exploration/explotation')
    parser.add_argument('--eps_end',
                    default=0.01,
                    type=float,
                    help='Smallest Allowable Epsilon value to be used for exploration/explotation')
    
    parser.add_argument('--mem_size',
                    default=30000,
                    type=int,
                    help='Size of the Replay Buffer')
    parser.add_argument('--n_games',
                    default=50_000,
                    type=int,
                    help='Maximum Number of Training Episodes Conducted')
    
    parser.add_argument('--batch_size',
                    default=128,
                    type=int,
                    help='Batch Size that will be taken from the Replay Buffer per training episode')
    
    parser.add_argument('--lr',
                    default=5e-3,
                    type=float,
                    help='Starting Learning Rate for the Network')
    
    parser.add_argument('--gamma',
                    default=0.1,
                    type=float,
                    help='Discount Factor for Future Rewards ')
    parser.add_argument('--Vol_Frac_1',
                    default=0.7,
                    type=float,
                    help='Volume Fraction during first progressive refinement')
    
    parser.add_argument('--Vol_Frac_2',
                    default=0.5,
                    type=float,
                    help='Final Volume Fraction ')
    
    parser.add_argument('--Vol_Frac_3',
                    default=0.25,
                    type=float,
                    help='Final Volume Fraction ')

    parser.add_argument('--SC',
                    default=10,
                    type=float,
                    help='Stress constraint, between 0 and 2 ')
    
    parser.add_argument('--P_Norm',
                    default=10,
                    type=int,
                    help='Smoothing Parameter for P-Norm Global Stress calculation')
    parser.add_argument('--filename_save',
                       default='DDQN_TopOpt_Generalized_CNN_4L_',
                       type=str,
                       help='When training, what name would you like your weights, and figure saved as')
    parser.add_argument('--filename_load',
                       default='DDQN_TopOpt_Generalized_CNN_4L_6by6',
                       type=str,
                       help='When testing, what name is your NN weights saved under')
    parser.add_argument('--Progressive_Refinement',
                       default=True,
                       type=bool)
    parser.add_argument('--LC',
                       default=False,
                       type=bool,
                       help="type in loading conditions manually")
    parser.add_argument('--Load_Checkpoints',
                       default=True,
                       type=bool,
                       help="Load Checkouts. ")
    
    parser.add_argument('--VF_S',
                       default=0,
                       type=int,
                       help="Use vol fraction constraint [0] or stress constraint [1]")
    parser.add_argument('--Time_Trial',
                       default=True,
                       type=bool,
                       help="Perform Time Trial")
    parser.add_argument('--configfile',
                       default='config.json',
                       type=str,
                       help="name of config file. ")
    parser.add_argument('--From_App',
                       default=True,
                       type=bool,
                       help="True if being called by an external app. Not sure this is needed. ")
    parser.add_argument('--base_folder',
                       default=r"PhDResearch/Top_Opt_RL/DQN",
                       type=str,
                       help="Folder where to find saved files. Helpful if not running the app from the main folder.  ")
 

    if args_in:
        # print(f"Using args {args_in}")
        # all_defaults = {}
        # for key in vars(args):
        #     all_defaults[key] = parser.get_default(key)
        args = parser.parse_args("")
    else: args = parser.parse_args()

    return args
    
