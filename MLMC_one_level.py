#Compute the mean of the solutions of the Heat Equation

from fenics import *

'''
Parameters :
    A sample of the random parameter
    Number of triangles in the mesh (param. of UnitSquare fct)
    Number of time steps
    Initial solution (for t=0)
'''

'''
Return :
    Time mean of the integrals of the solutions on the all domain
'''

def Level_solver(V,random_param, num_time_steps):
    time_steps_lenght = 1/(num_time_steps)
    integral = 0

    '''
    Initial solution
    '''
    mean = [0.5, 0.5]   #Mean of the Gaussian
    sig = 1             #Variance of the Gaussian

    init_fct = Expression('(4/sig - pow(2*(x[0]-m1)/sig,2) - pow(2*(x[1]-m2)/sig,2))*exp(-(pow(x[0]-m1,2) + pow(x[1]-m2,2))/sig)', sig=sig,
                          m1=mean[0], m2=mean[1], degree=1)

    u = TrialFunction(V)
    v = TestFunction(V)

    def boundary(x, on_boundary):   #Define the boundary
        return on_boundary
    u_D = Constant(0)   #Homogeneous Dirichlet Boundary Condition
    bc = DirichletBC(V, Constant(0), boundary) #Define the Dirichlet BC

    a = (u*v + Constant(random_param*time_steps_lenght)*dot(grad(u),grad(v)))*dx
    L = init_fct*v*dx
    u = Function(V)

    for k in range(num_time_steps-1):
        solve(a == L, u, bc)
        integral += assemble(u*dx)
        init_fct = u

    integral = integral/(num_time_steps)
    return integral
