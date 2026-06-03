import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# =====================================================================
# EXPERIMENT 1A: LAPLACE EQUATION WITH NEW BOUNDARIES (No Source)
# =====================================================================
def solve_exp_1a(h):
    n_intervals = int(1 / h)
    N = n_intervals - 1
    x = np.linspace(0, 1, n_intervals + 1)
    y = np.linspace(0, 1, n_intervals + 1)
    W = np.zeros((n_intervals + 1, n_intervals + 1))
    
    # NEW Boundary Conditions (Left/Right=100, Top/Bottom=0)
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
            
            # NO SOURCE TERM for 1a (Laplace Equation)
            
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

# =====================================================================
# EXPERIMENT 1B: POISSON EQUATION WITH ORIGINAL BOUNDARIES 
# =====================================================================
def solve_exp_1b(h):
    n_intervals = int(1 / h)
    N = n_intervals - 1
    x = np.linspace(0, 1, n_intervals + 1)
    y = np.linspace(0, 1, n_intervals + 1)
    W = np.zeros((n_intervals + 1, n_intervals + 1))
    
    # ORIGINAL Problem A Boundary Conditions
    for i in range(n_intervals + 1):
        W[0, i] = 0             # Left
        W[i, 0] = 0             # Bottom
        W[-1, i] = 100 * y[i]   # Right
        W[i, -1] = 100 * x[i]   # Top
        
    M = N * N
    A = lil_matrix((M, M))
    b = np.zeros(M)
    
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            k = (j - 1) * N + (i - 1)
            A[k, k] = 4
            
            # YES SOURCE TERM for 1b (Poisson Equation)
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

# =====================================================================
# MAIN EXECUTION, PLOTTING, AND TABLE GENERATION
# =====================================================================
if __name__ == "__main__":
    h_values = [0.25, 0.125, 0.0625]
    
    # --- Generate Data and Tables for 1a ---
    print("=======================================================")
    print("EXPERIMENT 1a: NEW BOUNDARIES (LAPLACE)")
    print("=======================================================")
    fig_1a = plt.figure(figsize=(18, 10))
    fig_1a.suptitle('Exp 1a: Laplace Equation with Modified Boundary Conditions', fontsize=16, fontweight='bold')
    
    for idx, h in enumerate(h_values):
        x_p, y_p, W_p = solve_exp_1a(h)
        mid_idx = int(0.5 / h)
        df_1a = pd.DataFrame({"x": x_p, f"u(x, y={y_p[mid_idx]:.3f})": W_p[:, mid_idx]})
        print(f"\n--- 1a Centerline Temperatures (h={h}, y=0.5) ---")
        print(df_1a.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        
        X_p, Y_p = np.meshgrid(x_p, y_p)
        
        # 1a 3D Plot
        ax3d = fig_1a.add_subplot(2, 3, idx + 1, projection='3d')
        ax3d.plot_surface(X_p, Y_p, W_p.T, cmap='viridis')
        ax3d.set_title(f'3D Surface (h = {h})')
        ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('Temperature')
        
        # 1a 2D Contour
        ax2d = fig_1a.add_subplot(2, 3, idx + 4)
        contour = ax2d.contourf(X_p, Y_p, W_p.T, levels=30, cmap='viridis')
        fig_1a.colorbar(contour, ax=ax2d)
        ax2d.set_title(f'Contour Map (h = {h})')
        ax2d.set_xlabel('x'); ax2d.set_ylabel('y')

    fig_1a.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    # --- Generate Data and Tables for 1b ---
    print("\n\n=======================================================")
    print("EXPERIMENT 1b: GAUSSIAN SOURCE (POISSON)")
    print("=======================================================")
    fig_1b = plt.figure(figsize=(18, 10))
    fig_1b.suptitle('Exp 1b: Poisson Equation with Gaussian Heat Source', fontsize=16, fontweight='bold')
    
    for idx, h in enumerate(h_values):
        x_p, y_p, W_p = solve_exp_1b(h)
        mid_idx = int(0.5 / h)
        df_1b = pd.DataFrame({"x": x_p, f"u(x, y={y_p[mid_idx]:.3f})": W_p[:, mid_idx]})
        print(f"\n--- 1b Centerline Temperatures (h={h}, y=0.5) ---")
        print(df_1b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        
        X_p, Y_p = np.meshgrid(x_p, y_p)
        
        # 1b 3D Plot
        ax3d = fig_1b.add_subplot(2, 3, idx + 1, projection='3d')
        ax3d.plot_surface(X_p, Y_p, W_p.T, cmap='inferno')
        ax3d.set_title(f'3D Surface (h = {h})')
        ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('Temperature')
        
        # 1b 2D Contour
        ax2d = fig_1b.add_subplot(2, 3, idx + 4)
        contour = ax2d.contourf(X_p, Y_p, W_p.T, levels=30, cmap='inferno')
        fig_1b.colorbar(contour, ax=ax2d)
        ax2d.set_title(f'Contour Map (h = {h})')
        ax2d.set_xlabel('x'); ax2d.set_ylabel('y')

    fig_1b.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.show() # Display both figures