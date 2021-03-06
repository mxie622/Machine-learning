# Ax = b
from numpy.linalg import *
from numpy import *

# A = array([[10., -1., 2., 0.],
#               [-1., 11., -1., 3.],
#               [2., -1., 10., -1.],
#               [0., 3., -1., 8.]])
# # initialize the RHS vector
# b = array([6., 25., -11., 15.])
# x = zeros((1, b.shape[0]))
# print(x)

A = array([[16, 3],
          [7, -11]])
b = array([[11],[13]])


def Gauss_Seidel(A, b, n, err):

    #        x1 = dot(T, x0) + C
    #        n = 10 #  iteration
    #        err = 1 # error acceptance

    if type(b) != ndarray and type(A) != ndarray:
        return "Change the input to array"
    else:
        L = tril(A)
        U = triu(A, k = 1)
        x = b # initialized

        T = -dot(inv(L), U)
        C = dot(inv(L), b)
        x = dot(T, x) + C
        i = 0

        last = x

        while True:
            x = dot(T, last) + C
            err = max(x - last)
            last = x
            print("Iteration {0}: {1}".format(i, x))
            i += 1
            if i == n or  err**2 < 0.0001:
                break
        return x,i

print(Gauss_Seidel(A=A, b=b, n = 30, err = 0.001)) # Approximate answer

print(dot(inv(A),b)) # True answer