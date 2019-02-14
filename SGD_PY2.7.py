#
# import pandas as pd
#
# import numpy as np
# data = [(1, 4,3, 12) , (1, 1,3, 22) ,(1,2.,3,31), (1,3.,2,12) , (1,4.,4,11),
#      (2, 5, 3, 13) , (4, 2, 1,22) ,(5, 4, 3,17), (1, 1,2, 13) , (1, 2, 4,14)]
# data = pd.DataFrame(data)
# #y[i] is the output of y = theta0 * x[0] + theta1 * x[1] +theta2 * x[2]
# #print(data.loc[0:len(data), [2, 3]])
#
# # print(len(data.iloc[1, :]))
# ncol = len(data.iloc[0, :])-1
#
# y = data[len(data.iloc[0, :])-1]
# x = data.iloc[0:len(data), 0:ncol]
# y = y.values
# x = x.values
# # x_list = []
# # print(type(y))
# # print(x)
# #
# #
# # print(type(x))
#
# # print(isinstance(data, list))
#
#
# # print(isinstance(y, list))
# # print(data[1])
# # print(len(x))
# #
# # # #
# def BGD(learning_rate, x, y):
#
#     error0 = 0
#     error1 = 0
# #    ncol = len(data.iloc[0, :]) - 1
#
#     # y = data[len(data.iloc[0, :]) - 1]
#     # x = data.iloc[0:len(data), 0:ncol]
#
# #   print(x)
#
#     epsilon = 0.0001
#     # learning rate
#     diff = [0, 0]
#
#     m = len(x)
#
#     # init the parameters to zero
#     theta0 = 0
#     theta1 = 0
#     theta2 = 0
#     # i = 1
#     # print(y[i] - (x[i][0] + theta1 * x[i][1] + theta2 * x[i][2]))
#     while True:
#
#     # calculate the parameters
#
#         for i in range(m - 1):
#             diff[0] =  y[i] - (x[i][0] + theta1 * x[i][1] + theta2 * x[i][2])
#             theta0 = theta0 + learning_rate * diff[0] * x[i][0]
#             theta1 = theta1 + learning_rate * diff[0] * x[i][1]
#             theta2 = theta2 + learning_rate * diff[0] * x[i][2]
#         for i in range(len(x)):
#             error1 += (y[i] - (x[i][0] + theta1 * x[i][1] + theta2 * x[i][2])) ** 2 / 2
#         if abs(error1 - error0) < epsilon:
#             break
#         else:
#             error0 = error1
#
#
#     return theta0, theta1, theta2, error1
#
# a = BGD(learning_rate=0.001, x = x, y = y)
# print(a)
#
#
#
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
# print(type(x))
# print(x)
# print(type(y))
# print(y)


def SGD(alpha, data):
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