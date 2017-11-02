from fenics import *
import numpy as np

# Determinist discretization variables

T = 1.0			# final time
time_steps = 30		# number of time steps
num_samples = 50	# number of samples for MC
Dt = T/time_steps	# time step size
J=0
sig = 1			# variance of gaussian init fct
mean = [0.5, 0.5]	# mean of gaussian init fct
psih=0			# value of the quantity of interest and after value of MC estimator of E
psih_2=0		# value of the square of the QOI and after valur of MC estimator of V
l=1

# Mesh, function space, trialfct and testfct

n=30
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, 'P', 1)
u = TrialFunction(V)
v = TestFunction(V)

# Boundary condition

def boundary(x, on_boundary):	# function to define the boundary
	return on_boundary

u_D = Constant(0)		# function for the homogeneous DirichletBC

bc = DirichletBC(V, u_D, boundary)	# define the DirichletBC

# Initial term u

u_n = Expression('(4/sig - pow(2*(x[0]-m1)/sig,2) - pow(2*(x[1]-m2)/sig,2))*exp(-(pow(x[0]-m1,2) + pow(x[1]-m2,2))/sig)', sig=sig, m1=mean[0], m2=mean[1], degree=1)

u_n = interpolate(u_n, V)		# interpolate u_n in the FE space V (compute the first solution u)

# /!\ We denote u_n the initial solution because we will use the same variable for the solution at each time step

# Bilinear form
a = (u*v + l*Dt*dot(grad(u),grad(v)))*dx

K=np.random.lognormal(0,1/8,num_samples)			# vector of num_samples samples of zero-mean LogNorm distrib with 1/8 standard deviation generated by numpy

for l in K:
	for k in range(time_steps-1):		# range(m) create a vector [0, 1, ..., m]
        L = u_n*v*dx
	    uh = Function(V)
	    solve(a == L, uh, bc)
        u_n.assign(uh)

# /!\ It is better to define a outside the loops and only change its parameter l in the loop than redefine a at each step because the definition of a need to call FFC just-in-time compiler and it takes tiiiiiime.

		# Compute the integral of u and sum it into J
		J+=assemble(u_n*dx)		# Approx the space integral and turn it into a number

		psih+=J

	psih_2+=((1/num_samples)*psih)**2

Varh=(1/(num_samples))*psih_2 - ((1/(num_samples))*psih)**2	# Standard MC to approx V

print((1/(num_samples))*psih)		# Print approx of E
print(Varh)						# Print approx of V
