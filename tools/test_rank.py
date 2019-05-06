import numpy as np

np.random.seed(712)

m = 8
n = 44

A_r = np.zeros((m, m, n))
A_t = np.zeros((n, m, n))

for i in range(m):
    for j in range(n):
        A_r[i, i, j] = 1

for i in range(n):
    for j in range(m):
        A_t[i, j, i] = 1

A = np.concatenate((A_r.reshape((m, m*n)), A_t.reshape((n, m*n))), axis=0)

print (np.linalg.matrix_rank(A))