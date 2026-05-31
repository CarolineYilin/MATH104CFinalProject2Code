import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# =====================================================================
# EXPERIMENT 1: POISSON EQUATION WITH SOURCE & NEW BOUNDARIES (1a & 1b)
# =====================================================================
def solve_poisson_custom(h):
    n_intervals = int(1 / h)
    N = n_intervals - 1
    x = np.linspace(0, 1, n_intervals + 1)
    y = np.linspace(0, 1, n_intervals + 1)
    W = np.zeros((n_intervals + 1, n_intervals + 1))
    
    # Boundary Conditions
    for i in range(n_intervals + 1):
        W[0, i] = 100   # Left
        W[-1, i] = 100  # Right
        W[i, 0] = 0     # Bottom
        W[i, -1] = 0    # Top
        
    M = N * N
    A = lil_matrix((M, M))
    b = np.zeros(M)
    
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            k = (j - 1) * N + (i - 1)
            A[k, k] = 4
            
            # Source Term f(x,y)
            x_val, y_val = x[i], y[j]
            f_xy = -5000 * np.exp(-((x_val - 0.5)**2 + (y_val - 0.5)**2) / 0.05)
            b[k] -= f_xy * (h**2) 
            
            # Neighbors
            if i > 1: A[k, k - 1] = -1
            else: b[k] += W[0, j]
                
            if i < N: A[k, k + 1] = -1
            else: b[k] += W[N + 1, j]
                
            if j > 1: A[k, k - N] = -1
            else: b[k] += W[i, 0]
                
            if j < N: A[k, k + N] = -1
            else: b[k] += W[i, N + 1]

    w_interior = spsolve(A.tocsr(), b)
    
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            k = (j - 1) * N + (i - 1)
            W[i, j] = w_interior[k]
            
    return x, y, W

if __name__ == "__main__":
    h_values = [0.25, 0.125, 0.0625]
    results = {}
    
    # Setup a large figure for the 2x3 grid
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Mesh Refinement Comparison for the Poisson Equation', fontsize=16, fontweight='bold')
    
    for idx, h in enumerate(h_values):
        print(f"=======================================================")
        print(f"EXPERIMENT 1: POISSON EQUATION (h = {h})")
        print(f"=======================================================")
        
        x_p, y_p, W_p = solve_poisson_custom(h)
        results[h] = (x_p, y_p, W_p)
        
        # Print Centerline Table
        mid_idx = int(0.5 / h)
        poisson_center_data = {"x": x_p, f"u(x, y={y_p[mid_idx]:.3f})": W_p[:, mid_idx]}
        df_poisson = pd.DataFrame(poisson_center_data)
        print("--- Centerline Temperatures (y=0.5) ---")
        print(df_poisson.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("\n")
        
        # --- Plotting ---
        X_p, Y_p = np.meshgrid(x_p, y_p)
        
        # Top Row: 3D Surface Plots
        ax3d = fig.add_subplot(2, 3, idx + 1, projection='3d')
        ax3d.plot_surface(X_p, Y_p, W_p.T, cmap='inferno')
        ax3d.set_title(f'3D Surface (h = {h})')
        ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('Temperature')
        
        # Bottom Row: 2D Contour Maps
        ax2d = fig.add_subplot(2, 3, idx + 4)
        contour = ax2d.contourf(X_p, Y_p, W_p.T, levels=30, cmap='inferno')
        fig.colorbar(contour, ax=ax2d)
        ax2d.set_title(f'Contour Map (h = {h})')
        ax2d.set_xlabel('x'); ax2d.set_ylabel('y')

    fig.tight_layout()
    plt.subplots_adjust(top=0.92) # Leave room for the main title
    plt.show()