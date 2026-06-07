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
            if i > 1: A[k, k - 1] = -1
            else: b[k] += W[0, j]
                
            # Right neighbor
            if i < N: A[k, k + 1] = -1
            else: b[k] += W[N + 1, j]
                
            # Bottom neighbor
            if j > 1: A[k, k - N] = -1
            else: b[k] += W[i, 0]
                
            # Top neighbor
            if j < N: A[k, k + N] = -1
            else: b[k] += W[i, N + 1]

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

# Data structures to save results
results_table = {
    "h": [],
    "Direct Time (s)": [],
    "GS Time (s)": [],
    "GS Iterations": []
}
max_errors = [] # Track max errors for line graph

# Create figures
fig_sol = plt.figure(figsize=(18, 12)) 
fig_err = plt.figure(figsize=(18, 5))  

for idx, h in enumerate(step_sizes):
    print(f"--- Solving for grid size h = {h} ---")
    
    # Solvers
    x, y, W_dir, t_dir = solve_direct(h)
    _, _, W_gs, iters, t_gs = solve_gauss_seidel(h)
    
    # Store metrics
    results_table["h"].append(h)
    results_table["Direct Time (s)"].append(f"{t_dir:.6f}")
    results_table["GS Time (s)"].append(f"{t_gs:.6f}")
    results_table["GS Iterations"].append(iters)

    # Calculate Exact Solution and Error
    X, Y = np.meshgrid(x, y)
    W_exact = 100 * X * Y
    
    abs_error_matrix = np.abs(W_dir - W_exact)
    max_abs_error = np.max(abs_error_matrix)
    max_errors.append(max_abs_error) # Save for plotting
    print(f"Max Absolute Error against Exact Solution u=100xy: {max_abs_error:.2e}")
    
    diff_matrix = np.abs(W_dir - W_gs)
    max_diff = np.max(diff_matrix)
    print(f"Max difference between Direct and GS: {max_diff:.2e}\n")

    # Plotting
    ax1 = fig_sol.add_subplot(2, 3, idx + 1, projection='3d')
    surf = ax1.plot_surface(X, Y, W_dir.T, cmap='viridis', edgecolor='none')
    ax1.set_title(f'3D Surface (h={h})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    
    ax2 = fig_sol.add_subplot(2, 3, idx + 4)
    contour_sol = ax2.contourf(X, Y, W_dir.T, levels=20, cmap='viridis')
    fig_sol.colorbar(contour_sol, ax=ax2)
    ax2.set_title(f'Solution Contour (h={h})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    ax3 = fig_err.add_subplot(1, 3, idx + 1)
    contour_err = ax3.contourf(X, Y, abs_error_matrix.T, levels=20, cmap='magma')
    fig_err.colorbar(contour_err, ax=ax3)
    ax3.set_title(f'Absolute Error Contour (h={h})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')

fig_sol.tight_layout()
fig_err.tight_layout()
plt.show()

# --- Plotting Mesh Refinement Behavior (Line Graphs) ---
fig_lines, (ax_err_line, ax_iter_line) = plt.subplots(1, 2, figsize=(14, 5))

# Error Behavior
ax_err_line.plot(step_sizes, max_errors, marker='o', color='red', linestyle='-')
ax_err_line.set_title('Error Behavior under Mesh Refinement')
ax_err_line.set_xlabel('Mesh Size (h)')
ax_err_line.set_ylabel('Max Absolute Error')
ax_err_line.grid(True)
ax_err_line.invert_xaxis()

# Iteration Count
ax_iter_line.plot(step_sizes, results_table["GS Iterations"], marker='o', color='blue', linestyle='-')
ax_iter_line.set_title('Gauss-Seidel Iteration Count under Mesh Refinement')
ax_iter_line.set_xlabel('Mesh Size (h)')
ax_iter_line.set_ylabel('Iterations Required to Converge')
ax_iter_line.grid(True)
ax_iter_line.invert_xaxis()

plt.tight_layout()
plt.show()

# --- Print Table 1: Computational Cost ---
print("\n=======================================================")
print("Table 1: Computational Cost Comparison")
print("=======================================================")
df_results = pd.DataFrame(results_table)
print(df_results.to_string(index=False))

# --- Print/Export Comprehensive Tables for ALL h ---
print("\n=======================================================")
print("Full Coordinate Tables: Exact vs Numerical vs Error")
print("=======================================================")

for h in step_sizes:
    print(f"\n--- Data Table for h = {h} ---")
    x, y, W_dir, _ = solve_direct(h)
    
    # Generate meshgrid to get exact coordinates
    X, Y = np.meshgrid(x, y)
    W_exact = 100 * X * Y
    
    # Flatten the 2D arrays into 1D columns for the table
    x_flat = X.flatten()
    y_flat = Y.flatten()
    exact_flat = W_exact.flatten()
    num_flat = W_dir.flatten()
    error_flat = np.abs(exact_flat - num_flat)
    
    # Create the DataFrame
    df_full = pd.DataFrame({
        "x": x_flat,
        "y": y_flat,
        "Exact u": exact_flat,
        "Numerical w": num_flat,
        "Abs Error": error_flat
    })
    
    # Print the table (Warning: Output is long for h=1/16)
    print(df_full.to_string(index=False, float_format=lambda v: f"{v:.4e}"))
    
    # Export to CSV for report inclusion (commented out to prevent automatic saving, uncomment if desired)
    filename = f"ProblemA_Table_h_{str(h).replace('/', '_')}.csv"
    df_full.to_csv(filename, index=False)
    print(f"-> Saved data to {filename}")