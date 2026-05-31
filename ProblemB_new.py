import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_banded

def solve_forward(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    w = np.sin(np.pi * x)
    
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
    x = np.linspace(0, L, N+1)
    w = np.sin(np.pi * x)
    
    main_diag = (1 + 2*lam) * np.ones(N-1)
    off_diag = -lam * np.ones(N-2)
    ab = np.zeros((3, N-1))
    ab[0, 1:] = off_diag  
    ab[1, :] = main_diag  
    ab[2, :-1] = off_diag 

    start = time.perf_counter()
    for j in range(1, M+1):
        b = w[1:N] 
        w[1:N] = solve_banded((1, 1), ab, b)
    elapsed = time.perf_counter() - start
    return w, elapsed

def solve_crank_nicolson(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    w = np.sin(np.pi * x)
    
    main_diag_A = (1 + lam) * np.ones(N-1)
    off_diag_A = (-lam / 2) * np.ones(N-2)
    ab = np.zeros((3, N-1))
    ab[0, 1:] = off_diag_A
    ab[1, :] = main_diag_A
    ab[2, :-1] = off_diag_A
    
    start = time.perf_counter()
    for j in range(1, M+1):
        # Vectorized RHS calculation for optimized performance
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
    print(f"Method Comparison | {case_name} Case (lambda={lam:.3f}, h={h:.3f})")
    print(f"=======================================================")
    
    table_sol = {"x": x, "Exact": u_exact, "Forward": w_fwd, "Backward": w_bwd, "Crank-Nic": w_cn}
    df_sol = pd.DataFrame(table_sol)
    print("\n--- Numerical Solutions ---")
    print(df_sol.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5e}"))
    
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
L, T, alpha = 1.0, 0.1, 1.0

# Loop through multiple spatial grids to fulfill project requirements
spatial_grids = [10, 20]

for N in spatial_grids:
    h = L / N
    
    # Calculate M to maintain lambda = 0.4 (Stable)
    k_stable = 0.4 * (h**2)
    M_stable = int(round(T / k_stable))
    
    # Calculate M to maintain lambda = 2.0 (Unstable)
    k_unstable = 2.0 * (h**2)
    M_unstable = int(round(T / k_unstable))

    # Run comparisons
    x, ex_s, fw_s, bw_s, cn_s, lam_s = run_comparison(L, T, alpha, N, M=M_stable, case_name="Stable")
    x, ex_u, fw_u, bw_u, cn_u, lam_u = run_comparison(L, T, alpha, N, M=M_unstable, case_name="Large Step (Unstable)")

    # ==========================================
    # FIGURE 1: NUMERICAL SOLUTIONS
    # ==========================================
    fig_sol, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Stable Solutions ---
    ax1.plot(x, ex_s, 'k-', linewidth=3, label='Exact')
    ax1.plot(x, fw_s, 'ro--', markersize=4, label='Forward', alpha=0.7)
    ax1.plot(x, bw_s, 'bs-.', markersize=4, label='Backward', alpha=0.7)
    ax1.plot(x, cn_s, 'g^:', markersize=4, label='Crank-Nicolson', alpha=0.7)
    ax1.set_title(f'Stable Case Solutions ($\\lambda = {lam_s:.2f}, h = {h:.3f}, k = {k_stable:.4f}$)')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Temperature u(x,T)')
    ax1.legend()
    ax1.grid(True)

    # --- Unstable Solutions ---
    ax2.plot(x, ex_u, 'k-', linewidth=3, label='Exact')
    fw_u_clipped = np.clip(fw_u, -0.2, 0.8) # Keeps the explosion from breaking the graph
    ax2.plot(x, fw_u_clipped, 'ro--', markersize=4, label='Forward (Clipped)', alpha=0.5)
    ax2.plot(x, bw_u, 'bs-.', markersize=4, label='Backward', alpha=0.7)
    ax2.plot(x, cn_u, 'g^:', markersize=4, label='Crank-Nicolson', alpha=0.7)
    
    if N == 10:
        dynamic_title = f'Unstable Regime: Error Not Yet Visible (5 steps)\n($\\lambda = {lam_u:.2f}, h = {h:.3f}, k = {k_unstable:.4f}$)'
    else:
        dynamic_title = f'Unstable Regime: Catastrophic Failure (20 steps)\n($\\lambda = {lam_u:.2f}, h = {h:.3f}, k = {k_unstable:.4f}$)'
        
    ax2.set_title(dynamic_title)
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Temperature u(x,T)')
    ax2.legend()
    ax2.grid(True)

    fig_sol.tight_layout()

    # ==========================================
    # FIGURE 2: ABSOLUTE ERRORS
    # ==========================================
    fig_err, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Stable Errors ---
    ax3.plot(x, np.abs(fw_s - ex_s), 'r--', linewidth=2, label='Forward Error')
    ax3.plot(x, np.abs(bw_s - ex_s), 'b-.', linewidth=2, label='Backward Error')
    ax3.plot(x, np.abs(cn_s - ex_s), 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson Error')
    ax3.set_title(f'Absolute Errors: Stable Case ($\\lambda = {lam_s:.2f}$)\nNotice CN is the most accurate')
    ax3.set_xlabel('Position (x)')
    ax3.set_ylabel('Absolute Error |u - w|')
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(0,0)) # Forces scientific notation
    ax3.legend()
    ax3.grid(True)

    # --- Unstable Errors ---
    # We clip the unstable forward error to 0.05, otherwise the explosion 
    # will make the y-axis so massive that you won't be able to see the Backward/CN errors!
    err_fw_u = np.clip(np.abs(fw_u - ex_u), 0, 0.05) 
    
    ax4.plot(x, err_fw_u, 'r--', linewidth=2, label='Forward Error (Exploding/Clipped)')
    ax4.plot(x, np.abs(bw_u - ex_u), 'b-.', linewidth=2, label='Backward Error')
    ax4.plot(x, np.abs(cn_u - ex_u), 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson Error')
    ax4.set_title(f'Absolute Errors: Unstable Case ($\\lambda = {lam_u:.2f}$)\nNotice Backward dampens more than CN')
    ax4.set_xlabel('Position (x)')
    ax4.set_ylabel('Absolute Error |u - w|')
    ax4.legend()
    ax4.grid(True)

    fig_err.tight_layout()
    
    # Show both figures at the same time
    plt.show()