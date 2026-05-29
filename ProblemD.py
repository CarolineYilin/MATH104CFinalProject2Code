import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.linalg import solve_banded

def exact_square_wave(x, t, alpha, terms=400):
    """
    Calculates the exact analytical solution for the discontinuous square wave 
    using a high-term Fourier Sine Series to minimize Gibbs phenomenon.
    """
    u = np.zeros_like(x)
    for n in range(1, terms + 1):
        Bn = (200 / (n * np.pi)) * (np.cos(n * np.pi * 0.4) - np.cos(n * np.pi * 0.6))
        u += Bn * np.sin(n * np.pi * x) * np.exp(-(n * np.pi * alpha)**2 * t)
    return u

def solve_forward(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    
    # 1e-5 tolerance ensures floating point math doesn't chop the edges
    w = np.where((x >= 0.4 - 1e-5) & (x <= 0.6 + 1e-5), 100.0, 0.0)
    
    for j in range(1, M+1):
        w_new = np.zeros(N+1)
        for i in range(1, N):
            w_new[i] = (1 - 2*lam)*w[i] + lam*(w[i+1] + w[i-1])
        w = w_new.copy()
    return w

def solve_backward(L, T, alpha, N, M):
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

def solve_crank_nicolson(L, T, alpha, N, M):
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
        b = np.zeros(N-1)
        for i in range(1, N):
            idx = i - 1
            b[idx] = (1 - lam)*w[i] + (lam/2)*(w[i+1] + w[i-1])
        w[1:N] = solve_banded((1, 1), ab, b)
    return w

def run_problem_D_experiment(L, T, alpha, N, M):
    h, k = L / N, T / M
    lam = (alpha**2 * k) / (h**2)
    x = np.linspace(0, L, N+1)
    
    w_fwd = solve_forward(L, T, alpha, N, M)
    w_bwd = solve_backward(L, T, alpha, N, M)
    w_cn = solve_crank_nicolson(L, T, alpha, N, M)
    u_exact = exact_square_wave(x, T, alpha)
    
    # Optional: Print tables to console
    df_err = pd.DataFrame({
        "x": x,
        "Err(Fwd)": np.abs(w_fwd - u_exact),
        "Err(Bwd)": np.abs(w_bwd - u_exact),
        "Err(CN)": np.abs(w_cn - u_exact)
    })
    print(df_err.iloc[1:N].to_string(index=False, float_format=lambda v: f"{v:.5e}"))
    
    return x, u_exact, w_fwd, w_bwd, w_cn, lam

# --- Execution & Plotting ---
L, T, alpha, N = 1.0, 0.005, 1.0, 40
M = 160 # High M ensures lambda is small and strictly stable

x, u_exact, fw, bw, cn, lam = run_problem_D_experiment(L, T, alpha, N, M)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Value Comparison
ax1.plot(x, u_exact, 'k-', linewidth=3, label='Exact (400-Term Fourier)')
ax1.plot(x, fw, 'r--', linewidth=2, label='Forward-Diff (1st Order)', alpha=0.8)
ax1.plot(x, bw, 'b-.', linewidth=2, label='Backward-Diff (1st Order)', alpha=0.8)
ax1.plot(x, cn, 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson (2nd Order)')
ax1.axhline(0, color='black', linestyle=':', alpha=0.5) # The 0-degree baseline
ax1.set_title(f'Problem D: Discontinuous Heat Shock ($T={T}$)\nNotice Crank-Nicolson dropping below 0°')
ax1.set_xlabel('Position (x)')
ax1.set_ylabel('Temperature (u)')
ax1.legend()
ax1.grid(True)

# Panel 2: Absolute Error
ax2.plot(x, np.abs(fw - u_exact), 'r--', linewidth=2, label='Forward Error')
ax2.plot(x, np.abs(bw - u_exact), 'b-.', linewidth=2, label='Backward Error')
ax2.plot(x, np.abs(cn - u_exact), 'g.-', markersize=8, linewidth=2, label='Crank-Nicolson Error')
ax2.set_title(f'Absolute Error Comparison\n(CN error spikes at the discontinuity)')
ax2.set_xlabel('Position (x)')
ax2.set_ylabel('Absolute Error |u - w|')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()