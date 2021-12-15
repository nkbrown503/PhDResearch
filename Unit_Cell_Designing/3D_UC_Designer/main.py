# -*- coding: utf-8 -*-



'''
This code was produced by Nathan Brown (12/15/2021) during his PhD research at Clemson University's
Mechanical Engineering Department


This code was built to randomly generate 3D unit-cells using cubic bezier curves
as the fundamental building tool. The code allows for the opportunity to mirror the designed 
unitcell to promote symmetry. The code will also allow a user to tesselate the unit-cell to produce
a lattice structure. Finally, the result of the design will be saved as a numpy file in a 'Design_Files' folder.'''


from Design_Functions_3D import Bezier_3D_UC_Builder, Mirror_Func, Tesselate_UC
from opts import parse_opts
import sys
opts=parse_opts()  
for It in range(0,opts.Iterations):
    sys.stdout.write('\rCurrently working on Iteration {}/{}...'.format(It+1,opts.Iterations))
    sys.stdout.flush()
    Element_Block=Bezier_3D_UC_Builder(opts.E_X,opts.E_Y,opts.E_Z,opts.Curves,opts.Iterations,opts.Mirror,opts.Type)
    Mirror_UC=Mirror_Func(opts.E_X,opts.E_Y,opts.E_Z,Element_Block,It,opts.Mirror,opts.Print_UC,opts.Save,opts.Tesselate)
    Tesselate_UC(opts.E_X,opts.E_Y,opts.E_Z,opts.UX,opts.UY,opts.UZ,It,Mirror_UC,opts.Type,opts.Save,opts.Tesselate,opts.Mirror)




 


