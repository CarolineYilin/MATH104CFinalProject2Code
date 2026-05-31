import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

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
            # Heat source centered at (0.5, 0.5)
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

# Execution and Plotting
h = 0.0625
x, y, W = solve_poisson_custom(h)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, W.T, cmap='inferno')
ax1.set_title('1(a) + 1(b): Poisson Eq with New Boundaries')
ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u(x,y)')

ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, W.T, levels=30, cmap='inferno')
fig.colorbar(contour, ax=ax2)
ax2.set_title('Contour Map of Heat Dispersion')

plt.tight_layout()
plt.show()
