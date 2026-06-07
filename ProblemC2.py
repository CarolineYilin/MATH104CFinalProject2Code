import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time  # <--- Moved to the top
from scipy.linalg import solve_banded

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
    L, T, alpha = 1.0, 0.005, 1.0
    
    # REQUIREMENT 1: Use multiple mesh sizes (h) AND time steps (k)
    # We pair each N with a different M to keep the simulation strictly stable
    test_cases = [(10, 10), (20, 40), (40, 160)]
    
    # Setup a large figure for a 2x3 grid (Solutions on top, Errors on bottom)
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(f'Mesh Refinement Analysis for Square Wave Heat Shock (T={T})', fontsize=16, fontweight='bold')
    
    for idx, (N, M) in enumerate(test_cases): 
        h = L / N
        k = T / M
        x_sq = np.linspace(0, L, N+1)
        u_exact = exact_square_wave(x_sq, T, alpha)
        
        # ==========================================
        # CALCULATE COMPUTATIONAL COST (TIMING)
        # ==========================================
        start_time = time.time()
        fw = solve_forward_sq(L, T, alpha, N, M)
        fw_time = time.time() - start_time

        start_time = time.time()
        bw = solve_backward_sq(L, T, alpha, N, M)
        bw_time = time.time() - start_time

        start_time = time.time()
        cn = solve_crank_nicolson_sq(L, T, alpha, N, M)
        cn_time = time.time() - start_time
        
        # ==========================================
        # PRINT TABLES TO CONSOLE (Requirement 5 & Cost)
        # ==========================================
        print(f"\n=======================================================")
        print(f"EXPERIMENT 2: SQUARE WAVE (N={N}, h={h:.3f})")
        print(f"=======================================================")
        
        # Print the newly calculated times so you can put them in your LaTeX report!
        print(f"Cost (s) -> Forward: {fw_time:.6f} | Backward: {bw_time:.6f} | Crank-Nic: {cn_time:.6f}")
        
        df_sq_sol = pd.DataFrame({
            "x": x_sq, "Exact": u_exact, "Forward": fw, "Backward": bw, "Crank-Nic": cn
        })
        print("\n--- Numerical Solutions ---")
        print(df_sq_sol.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5f}"))

        df_sq_err = pd.DataFrame({
            "x": x_sq,
            "Err(Fwd)": np.abs(fw - u_exact),
            "Err(Bwd)": np.abs(bw - u_exact),
            "Err(CN)": np.abs(cn - u_exact)
        })
        print("\n--- Absolute Errors ---")
        print(df_sq_err.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5e}"))
        
        # ==========================================
        # PLOT TO FIGURE (Requirement 6)
        # ==========================================
        ax_sol = axes[0, idx]
        ax_err = axes[1, idx]

        # --- Top Row: Solutions ---
        ax_sol.plot(x_sq, u_exact, 'k-', linewidth=3, label='Exact')
        ax_sol.plot(x_sq, fw, 'r--', linewidth=2, label='Forward', alpha=0.8)
        ax_sol.plot(x_sq, bw, 'b-.', linewidth=2, label='Backward', alpha=0.8)
        ax_sol.plot(x_sq, cn, 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson')
        ax_sol.axhline(0, color='black', linestyle=':', alpha=0.5) 
        
        ax_sol.set_title(f'Solutions (N = {N}, h = {h:.3f}, k = {k:.5f})')
        ax_sol.set_xlabel('Position (x)')
        ax_sol.set_ylabel('Temperature (u)')
        if idx == 0: ax_sol.legend()
        ax_sol.grid(True)

        # --- Bottom Row: Absolute Errors ---
        ax_err.plot(x_sq, np.abs(fw - u_exact), 'r--', linewidth=2, label='Forward Error')
        ax_err.plot(x_sq, np.abs(bw - u_exact), 'b-.', linewidth=2, label='Backward Error')
        ax_err.plot(x_sq, np.abs(cn - u_exact), 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson Error')
        
        ax_err.set_title(f'Absolute Errors (N = {N}, h = {h:.3f}, k = {k:.5f})')
        ax_err.set_xlabel('Position (x)')
        ax_err.set_ylabel('Absolute Error |u - w|')
        if idx == 0: ax_err.legend()
        ax_err.grid(True)
        
    fig2.tight_layout()
    plt.subplots_adjust(top=0.92) # Prevents graphs from overlapping the main title
    plt.show()