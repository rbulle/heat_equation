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

'''
MLMC variables
'''
num_level = int(input("Enter the number of levels (max 10):"))       #Number of levels
random_param = 0    #Init the random parameter
tri_mesh = [16*2**k for k in range(num_level)]
time_steps = [2*2**k for k in range(num_level)]
num_samples = [2**(10-k) for k in range(num_level)]

'''
Init result variables
'''
MLMC_estimator = 0      #Final estimator

'''
Parameters of the initial Gaussian
'''
mean = [0.5, 0.5]   #Mean of the Gaussian
sig = 1             #Variance of the Gaussian

'''
Finite Elements stuff
'''
meshes = [UnitSquareMesh(num_tri_mesh, num_tri_mesh) for num_tri_mesh in tri_mesh]       #Create a vector of meshes with increasing refinement
V = [FunctionSpace(mesh, 'P', 1) for mesh in meshes]

'''
Initial solution
'''
init_fct = Expression('(4/sig - pow(2*(x[0]-m1)/sig,2) - pow(2*(x[1]-m2)/sig,2))*exp(-(pow(x[0]-m1,2) + pow(x[1]-m2,2))/sig)', sig=sig, m1=mean[0], m2=mean[1], degree=1)


'''
Computation of the other levels
'''
for l in range(num_level):

    random_param = np.random.lognormal(0, 1/8, num_samples[l])

    for k in random_param:
        if l == 0:
            MLMC_estimator += (1/(num_samples[l]))*Level_solver(V[l], k, time_steps[l], init_fct)
        else:
            MLMC_estimator += (1/(num_samples[l]))*(Level_solver(V[l], k, time_steps[l], init_fct) - Level_solver(V[l-1], k, time_steps[l-1],
                                                                                                                  init_fct))

print(MLMC_estimator)
