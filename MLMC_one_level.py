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

def Level_solver(a, L, num_time_steps, num_tri_mesh):
    time_steps_lenght = 1/(num_time_steps)
    integral = 0

    mesh = UnitSquareMesh(num_tri_mesh, num_tri_mesh)
    V = FunctionSpace(mesh, 'P', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    def boundary(x, on_boundary):    #Define the square's boundary
       return on_boundary

    u_D = Constant(0)
    bc = DirichletBC(V, u_D, boundary)

    for k in range(num_time_steps-1):
        u = Function(V)
        solve(a == L, u, bc)
        integral += assemble(u*dx)
        init_fct = u

    integral = integral/(num_time_steps)
    return integral
