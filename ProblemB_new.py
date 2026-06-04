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
L, T, alpha = 1.0, 0.5, 1.0
spatial_grids = [10, 20]

# Define the regimes we want to test
regimes = {
    "Stable": 0.4,
    "Unstable": 2.0,
    "Massive": 10.0
}

def plot_regime_and_print_tables(regime_name, target_lambda, title_desc):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{regime_name} Regime Comparison | {title_desc}', fontsize=16, fontweight='bold')
    
    for col, N in enumerate(spatial_grids):
        h = L / N
        # Calculate M to maintain the target lambda for this specific N
        k_target = target_lambda * (h**2)
        M = max(1, int(round(T / k_target)))
        
        # Run comparison (this will print the tables to the console!)
        x, ex, fw, bw, cn, lam = run_comparison(L, T, alpha, N, M, f"{regime_name} (N={N})")
        
        # Top Row: Solutions
        axes[0, col].plot(x, ex, 'k-', linewidth=3, label='Exact')
        fw_clipped = np.clip(fw, -0.2, 0.8) # Keep explosions visible but contained
        axes[0, col].plot(x, fw_clipped, 'ro--', markersize=4, label='Forward (Clipped)', alpha=0.6)
        axes[0, col].plot(x, bw, 'bs-.', markersize=4, label='Backward', alpha=0.8)
        axes[0, col].plot(x, cn, 'g^:', markersize=4, label='Crank-Nicolson', alpha=0.8)
        axes[0, col].set_title(f'Solutions (h={h:.3f}, $\\lambda$={lam:.2f})')
        axes[0, col].set_xlabel('Position (x)'); axes[0, col].set_ylabel('Temperature')
        axes[0, col].legend(); axes[0, col].grid(True)
        
        # Bottom Row: Errors
        err_fw = np.clip(np.abs(fw - ex), 0, 0.05)
        axes[1, col].plot(x, err_fw, 'r--', linewidth=2, label='Forward Error')
        axes[1, col].plot(x, np.abs(bw - ex), 'b-.', linewidth=2, label='Backward Error')
        axes[1, col].plot(x, np.abs(cn - ex), 'g.-', markersize=6, linewidth=2, label='Crank-Nicolson Error')
        axes[1, col].set_title(f'Absolute Errors (h={h:.3f})')
        axes[1, col].set_xlabel('Position (x)'); axes[1, col].set_ylabel('Absolute Error')
        if regime_name == "Stable": axes[1, col].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axes[1, col].legend(); axes[1, col].grid(True)

    fig.tight_layout()
    plt.subplots_adjust(top=0.90)

# Generate everything grouped by Regime (Prints Tables -> Shows Figure)
plot_regime_and_print_tables("Stable", 0.4, "Demonstrating Convergence and Accuracy (Forward is Stable)")
plot_regime_and_print_tables("Unstable", 2.0, "Demonstrating Error Propagation (Forward Explodes on Fine Grid)")
plot_regime_and_print_tables("Massive", 10.0, "Demonstrating Implicit Stiffening (Backward Retains Artificial Heat)")

plt.show()