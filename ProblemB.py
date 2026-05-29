import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def solve_forward_difference(L, T, alpha, N, M):
    """
    Solves 1D Heat Equation u_t = alpha^2 * u_xx using Forward-Difference.
    """
    h = L / N       # Spatial step size
    k = T / M       # Time step size
    lam = (alpha**2 * k) / (h**2)
    
    # Grid arrays
    x = np.linspace(0, L, N+1)
    w = np.zeros(N+1)
    
    # Apply Initial Condition: u(x,0) = sin(pi * x)
    w = np.sin(np.pi * x)
    
    history = [w.copy()]
    
    # Time stepping loop
    for j in range(1, M+1):
        w_new = np.zeros(N+1)
        w_new[0] = 0.0 # Boundary u(0,t) = 0
        w_new[N] = 0.0 # Boundary u(L,t) = 0
        
        # Forward-Difference Stencil
        for i in range(1, N):
            w_new[i] = (1 - 2*lam)*w[i] + lam*(w[i+1] + w[i-1])
            
        w = w_new.copy()
        history.append(w.copy())
            
    return x, history, lam

def run_error_analysis(L, T, alpha, N, M, case_name):
    """
    Calculates exact solution, compares it to the numerical solution, 
    and generates the required tables and plots.
    """
    h = L / N
    k = T / M
    
    # 1. Get the numerical solution (we only need the final time step)
    x, history, lam = solve_forward_difference(L, T, alpha, N, M)
    w_num_final = history[-1] # The temperature distribution at t = T
    
    # 2. Calculate the Exact Solution: u(x,t) = e^(-alpha^2 * pi^2 * t) * sin(pi * x)
    u_exact_final = np.exp(-(alpha * np.pi)**2 * T) * np.sin(np.pi * x)
    
    # 3. Calculate Absolute Error
    Error = np.abs(w_num_final - u_exact_final)
    
    # --- Generate Table ---
    print(f"\n--- Heat Equation Data Table at t = {T} ({case_name} Case) ---")
    print(f"Parameters: h={h:.3f}, k={k:.4f}, lambda={lam:.3f}")
    table_data = {
        "x": x,
        "Numerical": w_num_final,
        "Exact": u_exact_final,
        "Abs Error": Error
    }
    df = pd.DataFrame(table_data)
    # Using scientific notation because the unstable error will be huge
    print(df.to_string(index=False, float_format=lambda v: f"{v:.6e}"))
    
    # --- Generate Comparison Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(x, u_exact_final, 'k-', linewidth=2, label='Exact Solution')
    
    # Use different colors to make it obvious
    color = 'blue' if case_name == 'Stable' else 'red'
    plt.plot(x, w_num_final, color=color, marker='o', linestyle='--', markersize=5, label=f'Numerical ({case_name})')
    
    # Fill the error gap
    plt.fill_between(x, u_exact_final, w_num_final, color=color, alpha=0.1, label='Error Gap')
    
    plt.title(f'Heat Equation: Exact vs Numerical at t={T}\n{case_name} Case ($\\lambda={lam:.3f}$)')
    plt.xlabel('Position (x)')
    plt.ylabel('Temperature (u)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Execution ---
L = 1.0       # Length of the rod
T = 0.1       # Final time
alpha = 1.0   # Thermal diffusivity
N = 20        # Spatial intervals (h = 0.05)

# Scenario 1: Stable (lambda <= 0.5)
# If h = 0.05, we need k <= 0.00125. 
# M = 100 gives k = 0.001 (lambda = 0.4)
run_error_analysis(L, T, alpha, N, M=100, case_name="Stable")

# Scenario 2: Unstable (lambda > 0.5)
# M = 20 gives k = 0.005 (lambda = 2.0)
run_error_analysis(L, T, alpha, N, M=20, case_name="Unstable")