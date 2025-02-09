# This is the .py file for Task 2 of Section 4

import numpy as np

def create_spin_matrices(N):
    up = np.array([1, 0])
    down = np.array([0, 1])
    S_plus = np.outer(down, up)
    S_minus = np.outer(up, down)
    Sz = np.array([[0.5, 0], [0, -0.5]])

    def extend_operator(S, position):
        I = np.eye(2)
        for i in range(N):
            if i == position:
                op = S
            else:
                op = I
            if i == 0:
                full_operator = op
            else:
                full_operator = np.kron(full_operator, op)
        return full_operator

    H = np.zeros((2**N, 2**N))
    for i in range(N):
        next_i = (i + 1) % N 
        H += 0.5 * (extend_operator(S_plus, i) @ extend_operator(S_minus, next_i) + 
                    extend_operator(S_minus, i) @ extend_operator(S_plus, next_i)) + \
             extend_operator(Sz, i) @ extend_operator(Sz, next_i)

    return H

# Because we must now consider both the number of spin states for an N spin system in addition to the 2^N 
# matrix complexity, our overall time complexity for the program becomes O(N2^N)

import time
import matplotlib.pyplot as plt

N_values = range(1, 10)  # Example range, adjust based on computational limits
execution_times = []

for N in N_values:
    start_time = time.time()
    H = create_spin_matrices(N)
    execution_times.append(time.time() - start_time)

plt.plot(N_values, execution_times, marker='o')
plt.xlabel('Number of Spins (N)')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time for Constructing Hamiltonian Matrices')
plt.yscale('log')  # Use logarithmic scale if needed to better visualize exponential growth
plt.show()
#plt.savefig('T2Q1_complexity.png')

# Question 2

from scipy.linalg import qr

def QR(H, max_iterations=100, tolerance=1e-10):
    for i in range(max_iterations):
        Q, R = qr(H)  
        H_new = R @ Q  
        if np.allclose(H, H_new, atol=tolerance):
            break
        H = H_new
    return np.diag(H)

# Question 3
# Luckily there already exist these decompositions in scipy libraries

# First here is the LU decomposition

from scipy.linalg import lu, inv

def LU(omega, H):
    n = H.shape[0]
    I = np.eye(n)  
    A = omega * I - H 
    P, L, U = lu(A)  
    LU = P @ L @ U  
    G = inv(LU) 
    return G

# And now for the Cholesky

from scipy.linalg import cholesky

def Cholesky(omega, H):
    n = H.shape[0]
    I = np.eye(n)
    A = omega * I - H 
    try:
        L = cholesky(A, lower=True)
        L_inv = inv(L)
        G = L_inv.T @ L_inv 
        return G
    except np.linalg.LinAlgError:
        return "matrix is not positive definite"
    
# Now to plot Green's function for N=30

# Update: my computer did NOT like that lol. Order of 2^30 is way too much for the computer I forgot
# that could be an issue. The suggestion that GPT has is to approixmate using Sparse matrix approach but
# even that requires explicit computation of the Hamiltonian. I will potentially come back to this but
# I do not see how the computation is realistic.

def green_fn(omega_values, N):
    greens = []
    H = create_spin_matrices(N)
    for i in omegas:
        G = LU(i, H)
        greens.append(np.trace(G))  # Example: plot the trace of G
        
    plt.figure()
    plt.plot(omega_values, greens, label='Trace of G')
    plt.xlabel('Frequency (ω)')
    plt.ylabel('Trace')
    plt.title('Green\'s Function Behavior')
    plt.legend()
    plt.show()
    plt.savefig('Green_plot.png')

omegas = np.linspace(0.1, 2, 100)

# Question 4


