import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

# =====================================================================
# EXPERIMENT 1: POISSON EQUATION WITH SOURCE & NEW BOUNDARIES (1a & 1b)
# =====================================================================
def solve_poisson_custom(h):
    n_intervals = int(1 / h)
    N = n_intervals - 1
    x = np.linspace(0, 1, n_intervals + 1)
    y = np.linspace(0, 1, n_intervals + 1)
    W = np.zeros((n_intervals + 1, n_intervals + 1))
    
    # --- PART 1(a): NEW BOUNDARY CONDITIONS ---
    # Left and Right edges = 100, Top and Bottom = 0
    for i in range(n_intervals + 1):
        W[0, i] = 100   # Left boundary (x=0)
        W[-1, i] = 100  # Right boundary (x=1)
        W[i, 0] = 0     # Bottom boundary (y=0)
        W[i, -1] = 0    # Top boundary (y=1)
        
    M = N * N
    A = lil_matrix((M, M))
    b = np.zeros(M)
    
    for j in range(1, N + 1):
        for i in range(1, N + 1):
            k = (j - 1) * N + (i - 1)
            A[k, k] = 4
            
            # --- PART 1(b): ADD SOURCE TERM f(x,y) ---
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
# EXPERIMENT 2: DIFFERENT INITIAL CONDITIONS (SQUARE WAVE) (Option 2)
# =====================================================================
def exact_square_wave(x, t, alpha, terms=400):
    u = np.zeros_like(x)
    for n in range(1, terms + 1):
        Bn = (200 / (n * np.pi)) * (np.cos(n * np.pi * 0.4) - np.cos(n * np.pi * 0.6))
        u += Bn * np.sin(n * np.pi * x) * np.exp(-(n * np.pi * alpha)**2 * t)
    return u

def solve_forward_sq(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    w = np.where((x >= 0.4 - 1e-5) & (x <= 0.6 + 1e-5), 100.0, 0.0)
    
    for j in range(1, M+1):
        w_new = np.zeros(N+1)
        for i in range(1, N):
            w_new[i] = (1 - 2*lam)*w[i] + lam*(w[i+1] + w[i-1])
        w = w_new.copy()
    return w

def solve_backward_sq(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    w = np.where((x >= 0.4 - 1e-5) & (x <= 0.6 + 1e-5), 100.0, 0.0)
    
    main_diag = (1 + 2*lam) * np.ones(N-1)
    off_diag = -lam * np.ones(N-2)
    ab = np.zeros((3, N-1))
    ab[0, 1:] = off_diag  
    ab[1, :] = main_diag  
    ab[2, :-1] = off_diag 

    for j in range(1, M+1):
        b = w[1:N] 
        w[1:N] = solve_banded((1, 1), ab, b)
    return w

def solve_crank_nicolson_sq(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    w = np.where((x >= 0.4 - 1e-5) & (x <= 0.6 + 1e-5), 100.0, 0.0)
    
    main_diag_A = (1 + lam) * np.ones(N-1)
    off_diag_A = (-lam / 2) * np.ones(N-2)
    ab = np.zeros((3, N-1))
    ab[0, 1:] = off_diag_A
    ab[1, :] = main_diag_A
    ab[2, :-1] = off_diag_A
    
    for j in range(1, M+1):
        # Vectorized RHS for speed
        b = (1 - lam)*w[1:N] + (lam/2)*(w[2:N+1] + w[0:N-1])
        w[1:N] = solve_banded((1, 1), ab, b)
    return w


# =====================================================================
# MAIN EXECUTION & PLOTTING
# =====================================================================

if __name__ == "__main__":
    # ---------------------------------------------------------
    # RUN EXPERIMENT 1: POISSON EQUATION
    # ---------------------------------------------------------
    h_poisson = 0.0625
    x_p, y_p, W_p = solve_poisson_custom(h_poisson)
    
    # Print Table for Poisson (Extracting the centerline y=0.5)
    print("=======================================================")
    print("EXPERIMENT 1: POISSON EQUATION CENTERLINE (y=0.5)")
    print("=======================================================")
    mid_idx = int(0.5 / h_poisson)
    poisson_center_data = {"x": x_p, f"u(x, y={y_p[mid_idx]:.3f})": W_p[:, mid_idx]}
    df_poisson = pd.DataFrame(poisson_center_data)
    print(df_poisson.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n")
    
    # Setup Poisson Figures
    X_p, Y_p = np.meshgrid(x_p, y_p)
    fig1 = plt.figure(figsize=(12, 5))
    ax1_1 = fig1.add_subplot(1, 2, 1, projection='3d')
    ax1_1.plot_surface(X_p, Y_p, W_p.T, cmap='inferno')
    ax1_1.set_title('1(a) + 1(b): Poisson Eq with New Boundaries')
    ax1_1.set_xlabel('x'); ax1_1.set_ylabel('y'); ax1_1.set_zlabel('u(x,y)')

    ax1_2 = fig1.add_subplot(1, 2, 2)
    contour = ax1_2.contourf(X_p, Y_p, W_p.T, levels=30, cmap='inferno')
    fig1.colorbar(contour, ax=ax1_2)
    ax1_2.set_title('Contour Map of Heat Dispersion')
    fig1.tight_layout()

    # ---------------------------------------------------------
    # RUN EXPERIMENT 2: SQUARE WAVE HEAT EQUATION
    # ---------------------------------------------------------
    L, T, alpha, N = 1.0, 0.005, 1.0, 40
    M = 160 # High M ensures lambda is small and strictly stable
    
    x_sq = np.linspace(0, L, N+1)
    u_exact = exact_square_wave(x_sq, T, alpha)
    fw = solve_forward_sq(L, T, alpha, N, M)
    bw = solve_backward_sq(L, T, alpha, N, M)
    cn = solve_crank_nicolson_sq(L, T, alpha, N, M)
    
    # Print Table for Square Wave
    print("=======================================================")
    print(f"EXPERIMENT 2: SQUARE WAVE ERROR TABLE (T={T})")
    print("=======================================================")
    df_sq_err = pd.DataFrame({
        "x": x_sq,
        "Err(Fwd)": np.abs(fw - u_exact),
        "Err(Bwd)": np.abs(bw - u_exact),
        "Err(CN)": np.abs(cn - u_exact)
    })
    print(df_sq_err.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5e}"))
    
    # Setup Square Wave Figures
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(15, 6))

    ax2_1.plot(x_sq, u_exact, 'k-', linewidth=3, label='Exact (400-Term Fourier)')
    ax2_1.plot(x_sq, fw, 'r--', linewidth=2, label='Forward-Diff (1st Order)', alpha=0.8)
    ax2_1.plot(x_sq, bw, 'b-.', linewidth=2, label='Backward-Diff (1st Order)', alpha=0.8)
    ax2_1.plot(x_sq, cn, 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson (2nd Order)')
    ax2_1.axhline(0, color='black', linestyle=':', alpha=0.5) 
    ax2_1.set_title(f'Experiment 2: Discontinuous Heat Shock ($T={T}$)\nNotice Crank-Nicolson dropping below 0°')
    ax2_1.set_xlabel('Position (x)')
    ax2_1.set_ylabel('Temperature (u)')
    ax2_1.legend()
    ax2_1.grid(True)

    ax2_2.plot(x_sq, np.abs(fw - u_exact), 'r--', linewidth=2, label='Forward Error')
    ax2_2.plot(x_sq, np.abs(bw - u_exact), 'b-.', linewidth=2, label='Backward Error')
    ax2_2.plot(x_sq, np.abs(cn - u_exact), 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson Error')
    ax2_2.set_title('Absolute Error Comparison\n(CN error spikes at the discontinuity)')
    ax2_2.set_xlabel('Position (x)')
    ax2_2.set_ylabel('Absolute Error |u - w|')
    ax2_2.legend()
    ax2_2.grid(True)
    fig2.tight_layout()

    # Show both sets of plots at once!
    plt.show()