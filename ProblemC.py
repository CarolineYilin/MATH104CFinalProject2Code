import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_banded

def solve_forward(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    w = np.sin(np.pi * np.linspace(0, L, N+1))
    
    start = time.perf_counter()
    for j in range(1, M+1):
        w_new = np.zeros(N+1)
        for i in range(1, N):
            w_new[i] = (1 - 2*lam)*w[i] + lam*(w[i+1] + w[i-1])
        w = w_new.copy()
    elapsed = time.perf_counter() - start
    return w, elapsed

def solve_backward(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    w = np.sin(np.pi * np.linspace(0, L, N+1))
    
    # Tridiagonal matrix setup for SciPy's solve_banded
    # size is (N-1) x (N-1) because boundaries are 0
    main_diag = (1 + 2*lam) * np.ones(N-1)
    off_diag = -lam * np.ones(N-2)
    ab = np.zeros((3, N-1))
    ab[0, 1:] = off_diag  # Upper diagonal
    ab[1, :] = main_diag  # Main diagonal
    ab[2, :-1] = off_diag # Lower diagonal

    start = time.perf_counter()
    for j in range(1, M+1):
        b = w[1:N] # Interior points
        # Solve A * w_new = w_old
        w[1:N] = solve_banded((1, 1), ab, b)
    elapsed = time.perf_counter() - start
    return w, elapsed

def solve_crank_nicolson(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    w = np.sin(np.pi * np.linspace(0, L, N+1))
    
    # LHS Matrix (A) setup for solve_banded
    main_diag_A = (1 + lam) * np.ones(N-1)
    off_diag_A = (-lam / 2) * np.ones(N-2)
    ab = np.zeros((3, N-1))
    ab[0, 1:] = off_diag_A
    ab[1, :] = main_diag_A
    ab[2, :-1] = off_diag_A
    
    start = time.perf_counter()
    for j in range(1, M+1):
        # FAST VECTORIZED RHS (B * w_old)
        # b[1:-1] = (1 - lam) * w[1:-1] + (lam / 2) * (w[2:] + w[:-2])
        b = (1 - lam) * w[1:N] + (lam / 2) * (w[2:N+1] + w[0:N-1])
            
        w[1:N] = solve_banded((1, 1), ab, b)
    elapsed = time.perf_counter() - start
    return w, elapsed

def run_comparison(L, T, alpha, N, M, case_name):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    
    u_exact = np.exp(-(alpha * np.pi)**2 * T) * np.sin(np.pi * x)
    
    w_fwd, t_fwd = solve_forward(L, T, alpha, N, M)
    w_bwd, t_bwd = solve_backward(L, T, alpha, N, M)
    w_cn, t_cn = solve_crank_nicolson(L, T, alpha, N, M)
    
    print(f"\n=======================================================")
    print(f"Method Comparison | {case_name} Case (lambda={lam:.3f})")
    print(f"=======================================================")
    
    # 1. Print Solutions Table (Ignoring boundaries x=0 and x=1 since they are always 0)
    table_sol = {
        "x": x,
        "Exact": u_exact,
        "Forward": w_fwd,
        "Backward": w_bwd,
        "Crank-Nic": w_cn
    }
    df_sol = pd.DataFrame(table_sol)
    print("\n--- Numerical Solutions ---")
    print(df_sol.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5e}"))
    
    # 2. Print Errors Table (Ignoring boundaries)
    table_err = {
        "x": x,
        "Err(Forward)": np.abs(w_fwd - u_exact),
        "Err(Backward)": np.abs(w_bwd - u_exact),
        "Err(Crank-Nic)": np.abs(w_cn - u_exact)
    }
    df_err = pd.DataFrame(table_err)
    print("\n--- Absolute Errors ---")
    print(df_err.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5e}"))
    
    print("\nComputational Cost (Seconds):")
    print(f"Forward: {t_fwd:.6f} | Backward: {t_bwd:.6f} | Crank-Nic: {t_cn:.6f}")
    
    return x, u_exact, w_fwd, w_bwd, w_cn, lam

# --- Execution & Plotting ---
L, T, alpha, N = 1.0, 0.1, 1.0, 20

# 1. Stable Case (lambda = 0.4)
x, ex_s, fw_s, bw_s, cn_s, lam_s = run_comparison(L, T, alpha, N, M=100, case_name="Stable")

# 2. Unstable / Large Time Step Case (lambda = 2.0)
x, ex_u, fw_u, bw_u, cn_u, lam_u = run_comparison(L, T, alpha, N, M=20, case_name="Large Step (Unstable)")

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Stable
ax1.plot(x, ex_s, 'k-', linewidth=3, label='Exact')
ax1.plot(x, fw_s, 'ro--', markersize=4, label='Forward', alpha=0.7)
ax1.plot(x, bw_s, 'bs-.', markersize=4, label='Backward', alpha=0.7)
ax1.plot(x, cn_s, 'g^:', markersize=4, label='Crank-Nicolson', alpha=0.7)
ax1.set_title(f'Stable Case ($\\lambda = {lam_s:.2f}$)')
ax1.set_ylabel('Temperature u(x,T)')
ax1.legend()
ax1.grid(True)

# Plot 2: Large Time Step
ax2.plot(x, ex_u, 'k-', linewidth=3, label='Exact')
# Clip forward difference so it doesn't ruin the y-axis scale when it blows up
fw_u_clipped = np.clip(fw_u, -0.2, 0.8) 
ax2.plot(x, fw_u_clipped, 'ro--', markersize=4, label='Forward (Exploded/Clipped)', alpha=0.5)
ax2.plot(x, bw_u, 'bs-.', markersize=4, label='Backward', alpha=0.7)
ax2.plot(x, cn_u, 'g^:', markersize=4, label='Crank-Nicolson', alpha=0.7)
ax2.set_title(f'Large Time Step ($\\lambda = {lam_u:.2f}$)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()