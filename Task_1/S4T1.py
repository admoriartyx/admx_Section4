# This is the .py file for Task 1 of Section 4
# I am separating all the tasks in anticipation of them being long sets of code

# Question 1
# Goal is to construct naive divide and conquer approach, then to find critical exponent using Masters
# Then compare results for time complexity

import numpy as np

# I chose to take the recursive route 

def matrix_multiplication(A, B):
    n = A.shape[0]
    if n == 1:
        return A * B
    else:
        # allocation of submatrices:
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]

        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]

        # now to multiply subs
        M1 = matrix_multiplication(A11, B11)
        M2 = matrix_multiplication(A12, B21)
        M3 = matrix_multiplication(A11, B12)
        M4 = matrix_multiplication(A12, B22)
        M5 = matrix_multiplication(A21, B11)
        M6 = matrix_multiplication(A22, B21)
        M7 = matrix_multiplication(A21, B12)
        M8 = matrix_multiplication(A22, B22)

        # net the results
        M12 = M1 + M2
        M34 = M3 + M4
        M56 = M5 + M6
        M78 = M7 + M8

        # final resultant matrix
        top = np.hstack((M12, M34))
        bottom = np.hstack((M56, M78))
        return np.vstack((top, bottom))
    

# Answering the second bullet point of the question, we recall the critical exponent can be found 
# as n = log_b(a) (where be is our base). Since the recursion formula has parameters b = 2 and a = 8,
# we find the critical exponent to be n = 3.

# Now for the third bullet point, we must compare graphically

import matplotlib.pyplot as plt
import time

sizes = [2, 4, 8, 16]  # Adjust for larger matrices if necessary
times = []

for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    start_time = time.time()
    C = matrix_multiplication(A, B)
    times.append(time.time() - start_time)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(sizes, times, 'o-', label='Experimental Time')
plt.plot(sizes, [x**3 / 100000 for x in sizes], '^-', label='Theoretical $n^3$ scale')
plt.xlabel('Matrix size (n)')
plt.ylabel('Time (s)')
plt.title('Time Complexity of Naive Matrix Multiplication')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('T1Q1_complexity.png')

# The resultant plot shows a convergence of the two data sets which not only surprised me but it also
# disagreed with the plot that my GPT prompt had originally generated. I inquired as to this difference
# and GPT explained to me that because we are measuring processing speeds, the trial is not independent
# of all other processes occurring on my device. The system is not isolated and therefore is impacted
# by other programs and processes that are being run simultaneously. However, some kind of convergence 
# between the two data sets is to be expected. 

# Question 2
# Implementing Strasser algorithm is very similar

def strassen(A, B):
    n = A.shape[0]
    if n == 1:
        return A * B
    else:
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]

        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]

        M1 = strassen(A11 + A22, B11 + B22)
        M2 = strassen(A21 + A22, B11)
        M3 = strassen(A11, B12 - B22)
        M4 = strassen(A22, B21 - B11)
        M5 = strassen(A11 + A12, B22)
        M6 = strassen(A21 - A11, B11 + B12)
        M7 = strassen(A12 - A22, B21 + B22)

        # Compute the submatrices of the product matrix C:
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6

        # Combine quadrants into a single result matrix
        top = np.hstack((C11, C12))
        bottom = np.hstack((C21, C22))
        return np.vstack((top, bottom))

# Because we are now working with one less M matrix our a = 7 and b = 2 so the critical exponent
# is about n = 2.8074

sizes = [2, 4, 8, 16]  # Adjust for larger matrices if necessary
times = []

for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    start_time = time.time()
    C = strassen(A, B)
    times.append(time.time() - start_time)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(sizes, times, 'o-', label='Experimental Time')
plt.plot(sizes, [x**2.8074 / 500000 for x in sizes], '^-', label='Theoretical $n^{2.8074}$ scale')
plt.xlabel('Matrix size (n)')
plt.ylabel('Time (s)')
plt.title('Time Complexity of Strassen\'s Matrix Multiplication')
plt.legend()
plt.grid(True)
plt.show()
#plt.savefig('T1Q1_strassen.png')

# This time the plot deems the Strassen algorithm far more efficient than the experimental time.
# There is no apparent convergence between the two data sets. Therefore I would say that the 
# asymptotic behaviors do not agree.

# This is the end of Task 1. I will create a new .py file for Task 2.






