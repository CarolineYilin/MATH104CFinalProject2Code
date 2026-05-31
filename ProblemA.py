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

# Create figures
fig = plt.figure(figsize=(18, 12))

for idx, h in enumerate(step_sizes):
    print(f"--- Solving for grid size h = {h} ---")
    
    # 1. Direct Solver
    x, y, W_dir, t_dir = solve_direct(h)
    
    # 2. Gauss-Seidel Solver
    _, _, W_gs, iters, t_gs = solve_gauss_seidel(h)
    
    # Store metrics
    results_table["h"].append(h)
    results_table["Direct Time (s)"].append(f"{t_dir:.6f}")
    results_table["GS Time (s)"].append(f"{t_gs:.6f}")
    results_table["GS Iterations"].append(iters)
    # --- NEW: Calculate Exact Solution and Error ---
    X, Y = np.meshgrid(x, y)
    W_exact = 100 * X * Y
    
    # Calculate absolute error for the interior points
    # (Excluding boundaries since error there is 0)
    abs_error_matrix = np.abs(W_dir - W_exact)
    max_abs_error = np.max(abs_error_matrix)
    print(f"Max Absolute Error against Exact Solution u=100xy: {max_abs_error:.2e}")
    
    # Calculate Max Difference between methods
    max_diff = np.max(np.abs(W_dir - W_gs))
    print(f"Max difference between Direct and GS: {max_diff:.2e}\n")

    # --- Plotting the Surface (Using Direct Solver Results) ---
    X, Y = np.meshgrid(x, y)
    
    # 3D Surface Plot
    ax1 = fig.add_subplot(2, 3, idx + 1, projection='3d')
    surf = ax1.plot_surface(X, Y, W_dir.T, cmap='viridis', edgecolor='none')
    ax1.set_title(f'3D Surface (h={h})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    
    # 2D Contour Plot
    ax2 = fig.add_subplot(2, 3, idx + 4)
    contour = ax2.contourf(X, Y, W_dir.T, levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax2)
    ax2.set_title(f'Contour Map (h={h})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

plt.tight_layout()
plt.show()

fig2, ax_line = plt.subplots(figsize=(8, 6))

for idx, h in enumerate(step_sizes):
    # Re-run the direct solver to get the data for this specific h
    x, y, W_dir, _ = solve_direct(h)
    
    # Find the index closest to y = 0.5
    y_mid_idx = int(0.5 / h) 
    
    # Plot the temperature profile along that slice
    ax_line.plot(x, W_dir[:, y_mid_idx], marker='o', label=f'h = {h}')

ax_line.set_title('Cross-Section Temperature Profile at y = 0.5')
ax_line.set_xlabel('x-coordinate')
ax_line.set_ylabel('Temperature u(x, 0.5)')
ax_line.legend()
ax_line.grid(True)
plt.show()

# --- Print Full Numerical Solution Tables for ALL h ---
print("\n=======================================================")
print("Full Numerical Solution Grids u(x,y)")
print("=======================================================")

for h in step_sizes:
    print(f"\n--- Numerical Solution Grid for h = {h} ---")
    # Re-run direct solver for the specific grid to get the matrix
    x_grid, y_grid, W_grid, _ = solve_direct(h)

    # Transpose so rows are 'y' and columns are 'x', then flip [::-1] 
    # so y=0 is at the bottom of the printed table, matching the physical geometry.
    W_display = W_grid.T[::-1]
    y_display = y_grid[::-1]

    df_grid = pd.DataFrame(
        W_display, 
        index=[f"y={yv:.4f}" for yv in y_display], 
        columns=[f"x={xv:.4f}" for xv in x_grid]
    )
    
    # Print the DataFrame. We use a slightly smaller float format 
    # to help the larger grids fit on screen.
    print(df_grid.to_string(float_format=lambda v: f"{v:.2f}"))


