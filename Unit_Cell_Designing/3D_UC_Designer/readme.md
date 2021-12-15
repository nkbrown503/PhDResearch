# 3D UnitCell Designer
## Created by Nathan Brown as part of his PhD Research at Clemson University

This code was built to randomly generate 3D unit-cells using cubic bezier curves
as the fundamental building tool. The code allows for the opportunity to mirror the designed 
unitcell to promote symmetry. The code will also allow a user to tesselate the unit-cell to produce
a lattice structure. Finally, the result of the design will be saved as a numpy file in a 'Design_Files' folder.


You can adjust any parameters within the opt.py file

Parameter    |      Default Setting               | Description 
-------------------------------------------------------------------------------------------------------------------------
E_X          |             20                     | How many elements in the X direction? This value is doubled if Mirror=True

E_Y          |             20                     | How many elements in the Y direction? This value is doubled if Mirror=True

E_Z          |             20                     | How many elements in the Z direction? This value is doubled if Mirror=True

UX           |              3                     | How many tesselations will the unit-cell undergo in the X direction?

UY           |              3                     | How many tesselations will the unit-cell undergo in the Y direction?

UZ           |              3                     |How many tesselations will the unit-cell undergo in the Z direction?

Curves       |              3                     | How many Bezier Curves will make up an individual unitcell? If Mirror=True this will define how many curves defined the                                                          original quater of the unitcell.

Iterations   |               1                    | How many unique unitcells designs do you want to produce?

Mirror       |              True                  |  Will the original design be mirrored about the X, Y, and Z axes. Mirror must be True to Tesselate

Type         |         'Corner' or 'Edge'         | 'Surface' will use the entire top and bottom 
                                                    surfaces as potential starting and ending points for the bezier curve. A random point
                                                    in the bottom surface will be used as the start point and a random point on the 
                                                     top surface will be used as the ending point of the curve. 
    
    
                                                    'Corner' will use the 8 cubic corners as potential starting and ending points for the 
                                                     Bezier curve. The unit-cell build will not terminate until each corner has been used
                                                     as a start or ending point at least once. This ensures that the build delivers a practical and 
                                                     feasible unit-cell solutions.
                                                     
Print_UC     |               True                |   Do you want to print a visual representation of the unitcell? This is represented as a scatter plot and can be changed to an                                                       alternative method

Save         |               True                |    Do you want to save the design as a numpy file to use in exterior applications?

Tesselate    |                False              |   Do you want to tesselate the original unit-cell in the X, Y, and Z directions according to UX,UY, and UZ?
