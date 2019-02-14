def Decomposition(M):
    import numpy as np
    if isinstance(M, np.matrix) or isinstance(M, np.array):

        U = np.linalg.eig(M * M.T) # U = M * M^T

        V = np.linalg.eig(M.T * M) # Vt = M^T * M

        U = U[1]

        sigma = U.I * M * V[1].I
    return U, sigma, V[1]


import numpy as np

a = Decomposition(M = np.mat([[3, 2, 2], [2, 3, -2]]))

print(a)
print(a[0] * a[1] * a[2])

# SVD decomposition





