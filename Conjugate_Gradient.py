from numpy.linalg import *

from numpy import *
A = array([[4, 1],
          [1, 3]])
b = array([[1],[2]])

# x0 = array([[2],[1]]) #
# r0 = b - dot(A, x0) #
# p0 = r0 #
# alpha0 = dot(transpose(r0), r0) / dot(dot(transpose(p0), A), p0) #
#
# x1 = x0 + alpha0 * p0
# r1 = r0 - alpha0 * dot(A, p0)
# beta0 = dot(transpose(r1), r1) / dot(transpose(r0), r0)
# p1 = r1 + beta0 * p0
# alpha1 = dot(transpose(r1), r1) / dot(dot(transpose(p1), A), p1)

n = 30

def CG(A, b, n):
    i = 0
    n = n
    x = b  #
    r = b - dot(A, x)  #
    p = r  #
    alpha = dot(transpose(r), r) / dot(dot(transpose(p), A), p)  #
    while True:
        last = r
        x = x + alpha * p
        r = r - alpha * dot(A, p)
        beta = dot(transpose(r), r) / dot(transpose(last), last)
        p = r + beta * p
        alpha = dot(transpose(r), r) / dot(dot(transpose(p), A), p)
        i += 1
        if i == n or max(last) ** 2 < 0.001:
            break

    return x

a = CG(A=A, b = b, n = n)

print(a)







