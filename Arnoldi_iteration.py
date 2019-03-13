# Arnoldi iteration: If A is Hermitian, Arnoldi ---> Lanczos
import numpy as np
def arnoldi_iteration(A, b, n):
    """
      Computes a basis of the (n+1)-Krylov subspace of A: the space
      spanned by {b, Ab, ..., A^n b}.

      Input
      A: mxm array
      b: initial vector (length m)
     n: dimension of Krylov subspace, must be >=1

     Returns Q, h
     Q: mx(n+1) array, the columns are an orthonormal basis of the
     Krylov subspace.
     h: (n+1)xn array, A on basis Q. It is upper Hessenberg.
     """
    m = A.shape[0]

    h = np.zeros((n+1, n))
    Q = np.zeros((m, n+1))

    q = b/np.linalg.norm(b)     # Normalize the input vector
    Q[:, 0] = q                 # Use it as the first Krylov vector

    for k in range(n):
        v = A.dot(q)            # Generate a new candidate vector
        for j in range(k+1):    # Subtract the projections on previous vectors
            h[j, k] = np.dot(Q[:, j].conj(),v)
            v = v - h[j, k]*Q[:, j]
        h[k+1, k] = np.linalg.norm(v)
        eps = 1e-12             # If v is shorter than this threshold it is the zero vector
        if h[k+1,k] > eps:      # Add the produced vector to the list, unless
            q = v / h[k+1, k]   #   the zero vector is produced.
            Q[:, k+1] = q
        else:                   # If that happens, stop iterating.
            return Q,h
        return Q,h


A = np.array([[16, 3],
          [7, -11]])
b = [11,13]

print(arnoldi_iteration(A=A, b = b, n = 1))



