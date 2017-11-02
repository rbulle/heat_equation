#Multi-level Monte Carlo for the Heat Equation in 2D on the unit square

from fenics import *
import numpy as np
from MLMC_one_level import Level_solver
from math import sqrt

'''
We will increase the number of time steps, the number of triangles at the edge
of the unit square and the number of samples in a geometric way, using the
following sequence : 2**l
In other terms, at each level we increase the number of time steps by
multiplying by 2 and the number of triangles in the mesh by multiplying by
sqrt(2).
In the same way we decrease the number of samples by dividing by 2 at each
level
'''

'''
MLMC constants
'''
l = 0               #Power coefficient to define geometric sequence
min_time_steps = 1  #Number min of time steps (for level 0)
                    #Time steps lenght is constant at each MC level
time_steps_lenght = 1/(2**l)
min_tri_mesh = 1    #Number minimum of triangles (level 0)
max_samples = 100   #Number maximum of samples (level 0)
num_level = 5       #Number of levels
random_param = 0    #Init the random parameter ~ LogNormal

'''
Init result variables
'''
MLMC_estimator = 0
increment_estimator = 0

'''
Parameters of the initial Gaussian
'''
mean = [0.5, 0.5]   #Mean of the Gaussian
sig = 1             #Variance of the Gaussian

'''
Finite Elements stuff
'''
mesh = UnitSquareMesh(min_tri_mesh, min_tri_mesh)
V = FunctionSpace(mesh, 'P', 1) #Lagrange P1 FE
u = TrialFunction(V)
v = TestFunction(V)

def boundary(x, on_boundary):   #Define the boundary
    return on_boundary
u_D = Constant(0)   #Homogeneous Dirichlet Boundary Condition
bc = DirichletBC(V, u_D, boundary) #Define the Dirichlet BC

'''
Initial solution
'''
u_n = Expression('(4/sig - pow(2*(x[0]-m1)/sig,2) - pow(2*(x[1]-m2)/sig,2))*exp(-(pow(x[0]-m1,2) + pow(x[1]-m2,2))/sig)', sig=sig, m1=mean[0], m2=mean[1], degree=1)

'''
Weak equation stuff
'''
a = (u*v + l*time_steps_lenght*dot(grad(u),grad(v)))*dx
L = u_n*v*dx

'''
Computation of level 0
'''
random_param = np.random.lognormal(0,1/8,max_samples)

for k in random_param:
    MLMC_estimator += Level_solver(k, min_time_steps, min_tri_mesh, u_n)

MLMC_estimator = (1/max_samples)*MLMC_estimator

'''
Computation of the other levels
'''
for l in range(1, num_level):
    samples_current = max_samples*(2**(-l))
    time_steps_current = min_time_steps*(2**l)
    tri_mesh_current = min_tri_mesh*(2**l)

    random_param = np.random.lognormal(0,1/8,max_samples)

    for k in random_param:
        increment_estimator += Level_solver(k, time_steps_current,
                                            tri_mesh_current, u_n)-Level_solver(k, int(time_steps_current/2), int(tri_mesh_current/2), u_n)
    MLMC_estimator += (1/samples_current)*increment_estimator

print(MLMC_estimator)
