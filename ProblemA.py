import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def setup_grid_and_boundaries(h):
    """Sets up the grid and applies the boundary conditions."""
    n_intervals = int(1 / h)
    N = n_intervals - 1  # Number of interior points in one dimension
    
    # Create 1D arrays for x and y
    x = np.linspace(0, 1, n_intervals + 1)
    y = np.linspace(0, 1, n_intervals + 1)
    
    # Initialize the full grid (including boundaries)
    W = np.zeros((n_intervals + 1, n_intervals + 1))
    
    # Apply Boundary Conditions
    # u(0, y) = 0  -> Left boundary (W[0, :])
    # u(x, 0) = 0  -> Bottom boundary (W[:, 0])
    # u(x, 1) = 100x -> Top boundary (W[:, -1])
    # u(1, y) = 100y -> Right boundary (W[-1, :])
    
    for i in range(n_intervals + 1):
        W[i, -1] = 100 * x[i]  # Top
        W[-1, i] = 100 * y[i]  # Right
        
    return x, y, W, N

def solve_direct(h):
    """Solves the Laplace equation using a Direct Sparse Matrix Solver."""
    start_time = time.time()
    x, y, W, N = setup_grid_and_boundaries(h)
    
    M = N * N # Total number of interior unknowns
    A = lil_matrix((M, M))
    b = np.zeros(M)
    
    # Build the matrix A and vector b
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            k = (j - 1) * N + (i - 1) # 1D index for the 2D grid
            
            A[k, k] = 4
            
            # Left neighbor
            if i > 1:
                A[k, k - 1] = -1
            else:
                b[k] += W[0, j] # Add left boundary to RHS
                
            # Right neighbor
            if i < N:
                A[k, k + 1] = -1
            else:
                b[k] += W[N + 1, j] # Add right boundary to RHS
                
            # Bottom neighbor
            if j > 1:
                A[k, k - N] = -1
            else:
                b[k] += W[i, 0] # Add bottom boundary to RHS
                
            # Top neighbor
            if j < N:
                A[k, k + N] = -1
            else:
                b[k] += W[i, N + 1] # Add top boundary to RHS

    # Solve the sparse linear system
    A_csr = A.tocsr()
    w_interior = spsolve(A_csr, b)
    
    # Map the 1D solution back to the 2D grid
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            k = (j - 1) * N + (i - 1)
            W[i, j] = w_interior[k]
            
    end_time = time.time()
    return x, y, W, end_time - start_time

def solve_gauss_seidel(h, tol=1e-5, max_iter=10000):
    """Solves the Laplace equation using Gauss-Seidel iteration."""
    start_time = time.time()
    x, y, W, N = setup_grid_and_boundaries(h)
    
    iters = 0
    error = float('inf')
    
    # Gauss-Seidel Loop
    while error > tol and iters < max_iter:
        max_diff = 0.0
        for i in range(1, N + 1):
            for j in range(1, N + 1):
                old_val = W[i, j]
                # 5-point stencil average
                W[i, j] = 0.25 * (W[i+1, j] + W[i-1, j] + W[i, j+1] + W[i, j-1])
                
                diff = abs(W[i, j] - old_val)
                if diff > max_diff:
                    max_diff = diff
                    
        error = max_diff
        iters += 1
        
    end_time = time.time()
    return x, y, W, iters, end_time - start_time

# --- Execution and Plotting ---
step_sizes = [1/4, 1/8, 1/16]

# Data structures to save results for your report
results_table = {
    "h": [],
    "Direct Time (s)": [],
    "GS Time (s)": [],
    "GS Iterations": []
}

# Create two separate figures
fig_sol = plt.figure(figsize=(18, 12)) # For Surfaces and Solutions
fig_err = plt.figure(figsize=(18, 5))  # Exclusively for Absolute Errors

for idx, h in enumerate(step_sizes):
    print(f"--- Solving for grid size h = {h} ---")
    
    # 1. Direct Solver
    x, y, W_dir, t_dir = solve_direct(h)
    
    # 2. Gauss-Seidel Solver
    _, _, W_gs, iters, t_gs = solve_gauss_seidel(h)
    
    # Store metrics for Table 1
    results_table["h"].append(h)
    results_table["Direct Time (s)"].append(f"{t_dir:.6f}")
    results_table["GS Time (s)"].append(f"{t_gs:.6f}")
    results_table["GS Iterations"].append(iters)

    # --- Calculate Exact Solution and Error ---
    X, Y = np.meshgrid(x, y)
    W_exact = 100 * X * Y
    
    # Calculate absolute error for the interior points
    abs_error_matrix = np.abs(W_dir - W_exact)
    max_abs_error = np.max(abs_error_matrix)
    print(f"Max Absolute Error against Exact Solution u=100xy: {max_abs_error:.2e}")
    
    # Calculate Max Difference between methods
    diff_matrix = np.abs(W_dir - W_gs)
    max_diff = np.max(diff_matrix)
    print(f"Max difference between Direct and GS: {max_diff:.2e}\n")

    # --- Plotting the Surface and Contours ---
    # 1. 3D Surface Plot (Top Row of Solution Figure)
    ax1 = fig_sol.add_subplot(2, 3, idx + 1, projection='3d')
    surf = ax1.plot_surface(X, Y, W_dir.T, cmap='viridis', edgecolor='none')
    ax1.set_title(f'3D Surface (h={h})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    
    # 2. 2D Contour Map of Solution (Bottom Row of Solution Figure)
    ax2 = fig_sol.add_subplot(2, 3, idx + 4)
    contour_sol = ax2.contourf(X, Y, W_dir.T, levels=20, cmap='viridis')
    fig_sol.colorbar(contour_sol, ax=ax2)
    ax2.set_title(f'Solution Contour (h={h})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    # 3. 2D Contour Map of Absolute Error (Placed in the separate Error Figure)
    ax3 = fig_err.add_subplot(1, 3, idx + 1)
    contour_err = ax3.contourf(X, Y, abs_error_matrix.T, levels=20, cmap='magma')
    fig_err.colorbar(contour_err, ax=ax3)
    ax3.set_title(f'Absolute Error Contour (h={h})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

fig_sol.tight_layout()
fig_err.tight_layout()
plt.show()

# --- Print Table 1: Computational Cost ---
print("\n=======================================================")
print("Table 1: Computational Cost Comparison")
print("=======================================================")
df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False))

# --- Print Exact vs Numerical vs Error Table for a Cross-Section ---
print("\n=======================================================")
print("Table: Exact vs Numerical vs Error for h = 0.25 at slice y = 0.5")
print("=======================================================")

h_table = 0.25
x_t, y_t, W_dir_t, _ = solve_direct(h_table)
_, _, W_gs_t, _, _ = solve_gauss_seidel(h_table)

# Find the index for the centerline y = 0.5
y_mid_idx = int(0.5 / h_table)

# Calculate exact solution at y = 0.5
exact_u = 100 * x_t * y_t[y_mid_idx]

# Create comparison DataFrame
df_error_table = pd.DataFrame({
    "x": x_t,
    "Exact u": exact_u,
    "Direct w": W_dir_t[:, y_mid_idx],
    "GS w": W_gs_t[:, y_mid_idx],
    "Error (Direct)": np.abs(W_dir_t[:, y_mid_idx] - exact_u),
    "Error (GS)": np.abs(W_gs_t[:, y_mid_idx] - exact_u)
})

print(df_error_table.to_string(index=False, float_format=lambda v: f"{v:.4e}"))

# --- Print Full Numerical Solution Tables for ALL h ---
print("\n=======================================================")
print("Full Numerical Solution Grids u(x,y)")
print("=======================================================")

for h in step_sizes:
    print(f"\n--- Numerical Solution Grid for h = {h} ---")
    x_grid, y_grid, W_grid, _ = solve_direct(h)

    # Transpose so rows are 'y' and columns are 'x', then flip [::-1] 
    # so y=0 is at the bottom of the printed table.
    W_display = W_grid.T[::-1]
    y_display = y_grid[::-1]

    df_grid = pd.DataFrame(
        W_display, 
        index=[f"y={yv:.4f}" for yv in y_display], 
        columns=[f"x={xv:.4f}" for xv in x_grid]
    )
    
    print(df_grid.to_string(float_format=lambda v: f"{v:.2f}"))