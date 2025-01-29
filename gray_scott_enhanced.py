#!/usr/bin/env python3
"""
Enhanced Gray-Scott Reaction-Diffusion Script

1) 2D Parameter Sweep (Finite-Difference)
2) 2D Finite-Element Solver in FEniCS
3) 3D Reaction-Diffusion (Finite-Difference)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

###############################################
# 1) 2D Parameter Sweep (Finite-Difference)
###############################################

def run_grayscott_2d_fd(F, k, steps=2000, size=200, Du=0.16, Dv=0.08, dt=1.0):
    """
    Runs the Gray-Scott reaction-diffusion model in 2D using a simple finite-difference Laplacian.
    
    Parameters
    ----------
    F : float
        Feed rate.
    k : float
        Kill rate.
    steps : int
        Number of time steps to simulate.
    size : int
        Size of the 2D grid (size x size).
    Du : float
        Diffusion coefficient for U.
    Dv : float
        Diffusion coefficient for V.
    dt : float
        Time step.
    
    Returns
    -------
    U, V : np.ndarray
        Final concentrations of U and V on the 2D grid.
    """
    # Initialize concentration grids
    U = np.ones((size, size), dtype=np.float64)
    V = np.zeros((size, size), dtype=np.float64)

    # Add a perturbation in the center
    r = size // 10  # region size
    cx, cy = size // 2, size // 2
    U[cx-r:cx+r, cy-r:cy+r] = 0.50
    V[cx-r:cx+r, cy-r:cy+r] = 0.25
    
    # Optional: add small random noise
    V += 0.05 * np.random.random((size, size))
    
    # Helper function for Laplacian using finite differences
    def laplacian(Z):
        return (
            np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
            np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
            4.0 * Z
        )
    
    for _ in range(steps):
        Ulap = laplacian(U)
        Vlap = laplacian(V)
        
        # Reaction term
        UVV = U * (V**2)
        
        # Update equations
        U += dt * (Du * Ulap - UVV + F * (1 - U))
        V += dt * (Dv * Vlap + UVV - (F + k) * V)
    
    return U, V


def parameter_sweep_2d():
    """
    Example parameter sweep for (F, k) on a 2D grid.
    Saves images of final U-patterns for each pair.
    """
    # Ranges for F and k
    Fs = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    ks = [0.04, 0.05, 0.06, 0.07]
    
    output_dir = "pattern_results_2d"
    os.makedirs(output_dir, exist_ok=True)
    
    for F in Fs:
        for K in ks:
            U_final, V_final = run_grayscott_2d_fd(F, K, steps=2000, size=200)
            
            plt.figure(figsize=(4,4))
            plt.imshow(U_final, cmap='inferno', vmin=0, vmax=1)
            plt.title(f"F={F}, k={K}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"pattern_F{F}_k{K}.png"))
            plt.close()
    
    print(f"Parameter sweep completed. Check the '{output_dir}' folder for results.")


###############################################
# 2) 2D Finite-Element Solver (FEniCS)
###############################################
# You need to install FEniCS before running this section.
###############################################
try:
    from fenics import *
    fenics_available = True
except ImportError:
    fenics_available = False

def run_grayscott_fenics_2d(Du=0.16, Dv=0.08, F=0.06, k=0.062, 
                            T=50.0, num_steps=500, nx=50, ny=50):
    """
    Runs the Gray-Scott model in 2D using the FEniCS finite-element library.
    
    Parameters
    ----------
    Du, Dv : float
        Diffusion coefficients for U and V.
    F, k : float
        Feed and kill rates.
    T : float
        Final time.
    num_steps : int
        Number of time steps.
    nx, ny : int
        Mesh resolution in x and y directions.
    
    Returns
    -------
    U_sol, V_sol : dolfin.functions.function.Function
        Final solutions for U and V in FEniCS function format.
    """
    if not fenics_available:
        raise RuntimeError("FEniCS is not installed. Please install it to run this function.")
    
    dt = T / num_steps
    
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx, ny)
    P1 = FiniteElement('P', mesh.ufl_cell(), 1)
    ME = FunctionSpace(mesh, P1 * P1)
    
    # Define initial conditions
    class InitialCondition(UserExpression):
        def eval(self, values, x):
            # U around 1, V around 0
            U_0 = 1.0
            V_0 = 0.0
            # Put a patch in the center
            if 0.4 < x[0] < 0.6 and 0.4 < x[1] < 0.6:
                U_0 = 0.5
                V_0 = 0.25
            values[0] = U_0
            values[1] = V_0
        
        def value_shape(self):
            return (2,)
    
    # Create initial condition
    u0 = InitialCondition(degree=1)
    u = Function(ME)
    u.interpolate(u0)
    
    # To store the previous solution
    u_n = Function(ME)
    u_n.assign(u)
    
    # Split mixed functions
    U, V = split(u)
    U_n, V_n = split(u_n)
    
    # Define test functions
    phiU, phiV = TestFunctions(ME)
    
    # Reaction terms
    UVV = U*V*V
    fU = -UVV + F*(1 - U)
    fV = +UVV - (F + k)*V
    
    # Weak formulation
    F1 = ( (U - U_n)/dt*phiU*dx
           + Du*dot(grad(U), grad(phiU))*dx
           - fU*phiU*dx )
    F2 = ( (V - V_n)/dt*phiV*dx
           + Dv*dot(grad(V), grad(phiV))*dx
           - fV*phiV*dx )
    
    F_total = F1 + F2
    problem = NonlinearVariationalProblem(F_total, u)
    solver  = NonlinearVariationalSolver(problem)
    
    # Solve time steps
    for _ in range(num_steps):
        solver.solve()
        u_n.assign(u)
    
    # Extract final solutions
    U_sol, V_sol = u.split(deepcopy=True)
    return U_sol, V_sol


def fenics_2d_demo():
    """
    Demonstration of using FEniCS to solve Gray-Scott in 2D.
    Plots the final U solution.
    """
    if not fenics_available:
        print("FEniCS not installed. Skipping this demo.")
        return
    
    # Example parameters
    Du, Dv = 0.16, 0.08
    F, k   = 0.060, 0.062
    U_sol, V_sol = run_grayscott_fenics_2d(Du, Dv, F, k, T=50, num_steps=300, nx=50, ny=50)
    
    # Convert FEniCS Function to NumPy array for plotting
    U_array = U_sol.compute_vertex_values()
    
    # Plot using matplotlib (Fenics also has a built-in plot, but let's do standard MPL)
    plt.figure()
    # We need the mesh's coordinates to reshape properly
    mesh = U_sol.function_space().mesh()
    coords = mesh.coordinates()
    
    # The shape might not be trivial, so let's do a quick trick:
    #  - We'll create an array of the same dimension as (nx+1) x (ny+1) if uniform mesh
    nx, ny = mesh.num_cells(), mesh.num_cells()
    # Actually, an easier approach is to use 'plot' from fenics directly:
    import matplotlib.tri as mtri
    
    triang = mtri.Triangulation(coords[:,0], coords[:,1], triangles=mesh.cells())
    plt.tricontourf(triang, U_array, cmap='inferno', levels=50)
    plt.colorbar()
    plt.title("Final U concentration (FEniCS 2D)")
    plt.axis('equal')
    plt.show()


###############################################
# 3) 3D Reaction-Diffusion (Finite-Difference)
###############################################
def run_grayscott_3d_fd(Du=0.16, Dv=0.08, F=0.06, k=0.062,
                        steps=200, size=64, dt=1.0):
    """
    Runs a 3D Gray-Scott model with finite-difference Laplacian.
    Returns the 3D arrays U, V.
    
    Parameters
    ----------
    Du, Dv : float
        Diffusion coefficients for U and V.
    F, k : float
        Feed and kill rates.
    steps : int
        Number of time steps.
    size : int
        Grid dimension (size x size x size).
    dt : float
        Time step.
    
    Returns
    -------
    U, V : 3D np.ndarray
        Final concentrations of U and V in 3D.
    """
    U = np.ones((size, size, size), dtype=np.float64)
    V = np.zeros((size, size, size), dtype=np.float64)

    # Perturbation in the middle
    r = size // 8
    cx = cy = cz = size // 2
    U[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 0.50
    V[cx-r:cx+r, cy-r:cy+r, cz-r:cz+r] = 0.25
    
    def laplacian_3d(A):
        return (
            np.roll(A, 1, axis=0) + np.roll(A, -1, axis=0) +
            np.roll(A, 1, axis=1) + np.roll(A, -1, axis=1) +
            np.roll(A, 1, axis=2) + np.roll(A, -1, axis=2) -
            6.0 * A
        )
    
    for _ in range(steps):
        Ulap = laplacian_3d(U)
        Vlap = laplacian_3d(V)
        UVV  = U * (V**2)
        
        U += dt * (Du * Ulap - UVV + F*(1 - U))
        V += dt * (Dv * Vlap + UVV - (F + k)*V)
    
    return U, V


def demo_3d():
    """
    Demonstrates a 3D Gray-Scott simulation and shows how you might visualize it.
    """
    print("Running a 3D Gray-Scott simulation (this might take a bit)...")
    U_3d, V_3d = run_grayscott_3d_fd(steps=200, size=64)

    # If you have Mayavi or pyVista installed, you can do 3D volume/isosurface rendering.
    # For example, with Mayavi:
    """
    from mayavi import mlab
    mlab.contour3d(U_3d, contours=8, transparent=True)
    mlab.title("3D Gray-Scott: U concentration")
    mlab.show()
    """
    # Alternatively, you can save slices as images:
    slice_z = U_3d.shape[2] // 2
    plt.figure()
    plt.imshow(U_3d[:,:,slice_z], cmap='inferno', origin='lower', vmin=0, vmax=1)
    plt.title("Central slice of U in 3D")
    plt.colorbar()
    plt.show()
    print("3D demo complete.")


###############################################
# MAIN DEMO
###############################################
if __name__ == "__main__":
    # 1) Parameter Sweep in 2D
    print("=== 2D Parameter Sweep (Finite-Difference) ===")
    parameter_sweep_2d()
    
    # 2) FEniCS 2D
    print("\n=== 2D FEniCS Solver ===")
    fenics_2d_demo()
    
    # 3) 3D Reaction-Diffusion
    print("\n=== 3D Gray-Scott (Finite-Difference) ===")
    demo_3d()
