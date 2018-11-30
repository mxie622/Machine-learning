from math import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pca(input_data, percent = 0.95):
    mean_each_variable = np.mean(input_data,axis = 0) # axis = 0 for row，1 for column）
    newData = input_data - mean_each_variable

    covariance_matrix=np.cov(newData,rowvar=0) # rowvar = 0 indicates a row represent a sample
    print(covariance_matrix)
    eigen_values,eigen_vectors = np.linalg.eig(np.mat(covariance_matrix))
    print(eigen_values)
    print(eigen_vectors)
    n=percentage2n(eigen_values, percent)          # For getting percent, need n of eigenvalues

    eigValIndice=np.argsort(eigen_values)            # Sort eigenvalues ascending
    n_eigValIndice=eigValIndice[-1:-(n+1):-1]
    n_eigen_vectors=eigen_vectors[:,n_eigValIndice]
    lowDinput_data=newData * n_eigen_vectors               # data in low dimension
    reconMat=(lowDinput_data * n_eigen_vectors.T) + mean_each_variable
    return reconMat,lowDinput_data,n

def percentage2n(eigen_values,percentage):
    sortArray=np.sort(eigen_values)   # Ascending
    sortArray=sortArray[-1::-1]  # Descending
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

if __name__ == '__main__':
    data = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9, 35],
                  [42, 45, 11],
                  [9, 48, 5],
                  [11, 21, 14],
                  [8, 5, 15],
                  [11, 12, 21],
                  [21, 20, 25]])

    fig = plt.figure()
    ax = plt.subplot(111,projection='3d')
    ax.scatter(data[0],data[1],data[2],c='y')
    ax.set_zlabel('Z') # Axis
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
#    plt.show()
    print(data)
    fin = pca(data,0.9)
    mat =fin[1]
    print(mat)
    ax.scatter(mat[0],mat[1],mat[2],c='y')
    # plt.show()
