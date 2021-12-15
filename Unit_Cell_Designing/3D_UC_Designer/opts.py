# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 10:17:32 2021

@author: nbrow
"""


import argparse 
def parse_opts(args_in = None):
    parser=argparse.ArgumentParser()
    ''' Parameters invovled with generic environment building'''
    parser.add_argument('--E_X',
                        default=20,
                        type=int,
                        help='Number of X Elements (Before Mirroring)')
    
    parser.add_argument('--E_Y',
                        default=20,
                        type=int,
                        help='Number of Y Elements (Before Mirroring)')
    parser.add_argument('--E_Z',
                        default=20,
                        type=int,
                        help='Number of Z Elements (Before Mirroring)')
    parser.add_argument('--UX',
                        default=3,
                        type=int,
                        help='Number of X Tesselations')
    parser.add_argument('--UY',
                        default=3,
                        type=int,
                        help='Number of Y Tesselations')
    parser.add_argument('--UZ',
                        default=3,
                        type=int,
                        help='Number of Z Tesselations')
    parser.add_argument('--Curves',
                        default=3,
                        type=int,
                        help='How many bezier curves do you want to produce? This variable is more practical for the Type `Edge`')
    parser.add_argument('--Iterations',
                        default=5,
                        type=int,
                        help='How many unitcells would you like to produce?')
    
    parser.add_argument('--Mirror',
                        default=True,
                        type=bool,
                        help='If True, the bezier curve design will be mirrored about the X,Y, and Z axes. If false, there will be no mirroring ')
    
    parser.add_argument('--Type',
                        default='Corner',
                        type=str,
                        help='Corner: The 8 corners of a cube will act as the starting and ending points of the bezier curve.     Surface: Any point in the top and bottom surfaces can act as the starting and ending points of the bezier curves')
    
    parser.add_argument('--Print_UC',
                        default=True,
                        type=bool,
                        help='True: Print a 2x2 subplot of the designed UC   False: Dont Print')

    parser.add_argument('--Save',
                        default=True,
                        type=bool,
                        help='True: Save the UC design as a numpy file   False: Dont Save')

    parser.add_argument('--Tesselate',
                        default=False,
                        type=bool,
                        help='True: Tesselate the unitcell according the variables UX, UY, and UZ for the x,y, and z directions   False: Dont Tesselate and Save the individual unitcells')


    if args_in:
        # print(f"Using args {args_in}")
        # all_defaults = {}
        # for key in vars(args):
        #     all_defaults[key] = parser.get_default(key)
        args = parser.parse_args("")
    else: args = parser.parse_args()

    return args
    
