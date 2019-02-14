#
#
import pandas as pd

import numpy as np
data = [(1, 4,3, 12) , (1, 1,3, 22) ,(1,2.,3,31), (1,3.,2,12) , (1,4.,4,11),
     (2, 5, 3, 13) , (4, 2, 1,22) ,(5, 4, 3,17), (1, 1,2, 13) , (1, 2, 4,14)]
data = pd.DataFrame(data)
#y[i] is the output of y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
#print(data.loc[0:len(data), [2, 3]])

# print(len(data.iloc[1, :]))
ncol = len(data.iloc[0, :])-1

y = data[len(data.iloc[0, :])-1]
y = y.values
x = data.iloc[0:len(data), 0:ncol]
x = x.values
print(type(x))
print(x)
print(type(y))
print(y)


def SGD(alpha, data):
    if isinstance(data, list):
        data = pd.DataFrame(data)

        ncol = len(data.iloc[0, :]) - 1

        y = data[len(data.iloc[0, :]) - 1]
        y = y.values
        x = data.iloc[0:len(data), 0:ncol]
        x = x.values

    epsilon = 0.0001
    # learning rate
#    alpha = 0.01
    diff = [0, 0]
    error1 = 0
    error0 = 0
    m = len(x)

    # init the parameters to zero
    theta0 = 0
    theta1 = 0
    theta2 = 0

    while True:

        # calculate the parameters
        for i in range(m):
            diff[0] = y[i] - (theta0 + theta1 * x[i][1] + theta2 * x[i][2])

            theta0 = theta0 + alpha * diff[0] * x[i][0]
            theta1 = theta1 + alpha * diff[0] * x[i][1]
            theta2 = theta2 + alpha * diff[0] * x[i][2]

        # calculate the cost function
        error1 = 0
        for lp in range(len(x)):
            error1 += (y[i] - (theta0 + theta1 * x[i][1] + theta2 * x[i][2])) ** 2 / 2

        if abs(error1 - error0) < epsilon:
            break
        else:
            error0 = error1
            #    print ' theta0 : %f, theta1 : %f, theta2 : %f, error1 : %f'%(theta0,theta1,theta2,error1)

    return theta0, theta1, theta2

a = SGD(0.005, data = data)
print(a)