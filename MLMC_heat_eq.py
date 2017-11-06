#Multi-level Monte Carlo for the Heat Equation in 2D on the unit square

from fenics import *
import numpy as np
from MLMC_one_level import Level_solver
from math import sqrt

'''
At each level we increase the number of time steps and the number of triangles at the edge of the square by
multiplying them by 2.
In the same way we decrease the number of samples by dividing by 2 at each
level.
We start with 2 time steps, 16 triangles at the edge and 2**10 samples.
'''

results = open("MLMC_results.txt", "a")     #Create a file.txt in "add" mode

'''
MLMC variables
'''
num_level = int(input("Enter the number of levels (max 10):"))
random_param = 0    #Init the random parameter
tri_mesh = [8*2**k for k in range(num_level)]      #Start with a mesh of 8*8*2 triangles
time_steps = [2*2**k for k in range(num_level)]     #Start with 2 time steps
num_samples = [2**(10-k) for k in range(num_level)] #Start with 2**10 samples

'''
Init result variables
'''
MLMC_estimator = 0      #Final estimator

'''
Finite Elements stuff
'''
meshes = [UnitSquareMesh(num_tri_mesh, num_tri_mesh) for num_tri_mesh in tri_mesh]       #Create a vector of meshes with increasing refinement
V = [FunctionSpace(mesh, 'P', 1) for mesh in meshes]        #Vector of corresponding discrete functions spaces

'''
Computation of MLMC estimator
'''
for l in range(num_level):

    random_param = np.random.lognormal(0, 1/8, num_samples[l])

    for k in random_param:
        if l == 0:          #The level 0 is different from the next ones
            MLMC_estimator += (1/(num_samples[l]))*Level_solver(V[l], k, time_steps[l])
        else:
            MLMC_estimator += (1/(num_samples[l]))*(Level_solver(V[l], k, time_steps[l]) - Level_solver(V[l-1], k, time_steps[l-1]))

results.write(str(num_level))
results.write('\t \t')
results.write(str(MLMC_estimator))
results.write('\n')
results.close()

print(MLMC_estimator)
