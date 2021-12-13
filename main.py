import numpy as np
import scipy.interpolate
import random as rn
import matplotlib.pyplot as plt
import math


def func(x):
    return np.sin(x) / x


def line_from_2_points(x1, y1, x2, y2, x):
    k = (y1 - y2) / (x1 - x2)
    b = y2 - k * x2
    return k * x + b


def noised_func(x):
    tmp = []
    for i in x:
        k = (1 + rn.randint(-10, 10) / 100)
        tmp.append(func(i) * k)
    return tmp


def piecewise_func(y, x, step):
    tmp = []
    for i in range(len(x) - 1):
        for j in np.arange(x[i], x[i + 1], step):
            tmp.append(line_from_2_points(x[i], y[i], x[i + 1], y[i + 1], j))
    return tmp


def fi1(x, x1, x2, x3):
    return (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))


def fi2(x, x1, x2, x3):
    return (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))


def fi3(x, x1, x2, x3):
    return (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2))


def lagrange(x1, y1, x2, y2, x3, y3, x):
    return y1 * fi1(x, x1, x2, x3) + y2 * fi2(x, x1, x2, x3) + y3 * fi3(x, x1, x2, x3)


def lagrange_polynomial(y, x, step):
    tmp = []
    arg = []
    for i in range(1, len(x) - 1):
        for j in np.arange(x[i - 1], x[i + 1], step):
            tmp.append(lagrange(x[i - 1], y[i - 1], x[i], y[i], x[i + 1], y[i + 1], j))
            arg.append(j)
    return arg, tmp


def aver(x, y):
    ind = 0
    tmp_y = []
    tmp_x = []
    for i in range(1, len(x)):
        if x[i] != x[ind]:
            if i - ind > 1:
                tmp_y.append(sum(y[ind:i - 1]) / (i - 1 - ind))
                tmp_x.append(x[ind])
                ind = i
        else:
            tmp_x.append(x[i])
            tmp_y.append(y[i])

    return tmp_x, tmp_y


def smooth(x, y):
    z = zip(x, y)
    zs = sorted(z, key=lambda tup: tup[0])
    x1 = [z[0] for z in zs]
    y1 = [z[1] for z in zs]
    x1, y1 = aver(x1, y1)
    return x1, y1


def piecewise_plot(x, y):
    size = int(len(x) / (len(x) - 2))
    for i in range(0, len(x), size):
        plt.plot(x[i:i + size], y[i:i + size])


def search_square_polynomial_coefficients(x, y, m):
    xy = []
    for k in range(len(x)):
        xy.append([x[k], y[k]])
    matrix, r_side_of_matrix = make_system(xy, m)
    cPoly = np.linalg.solve(matrix, r_side_of_matrix)
    return cPoly


def make_system(xy_list, basis):
    matrix = [[0] * basis for _ in range(basis)]
    right_side_of_matrix = [0] * basis
    for i in range(basis):
        for j in range(basis):
            sumA, sumB = 0, 0
            for k in range(len(xy_list)):
                sumA += xy_list[k][0] ** (i + j)
                sumB += xy_list[k][1] * xy_list[k][0] ** i
            matrix[i][j] = sumA
            right_side_of_matrix[i] = sumB
    return matrix, right_side_of_matrix


def create_square_polynomial(c, x):
    polynomial = 0
    for i in range(len(c)):
        polynomial += c[i] * x ** i
    return polynomial


def error(x, y, y2, str):
    sum = 0
    err = []
    for i in range(len(x)):
        sum += math.fabs(y2[i] - y[i]) ** 2
        err.append(math.fabs(y2[i] - y[i]) ** 2)
    sum /= len(x)
    sum = sum ** 0.5
    print("Average quad error:", sum)
    plt.plot(x, err, label=str)
    plt.grid()
    plt.legend()
    plt.show()


A = 0.3
B = 10.3
STEP = 0.5

X = np.arange(A, B, 0.01)
args = np.arange(A, B + STEP, STEP)
arr = noised_func(args)
args_piecewise, arr_piecewise_linear = lagrange_polynomial(arr, args, 0.01)

plt.plot(args, func(args), label='piecewise original line')
plt.plot(X, func(X), label='function')
plt.scatter(args, arr, color="red", label='experimental points')
piecewise_plot(args_piecewise, arr_piecewise_linear)
plt.grid()
plt.legend()
plt.show()

smt_x, smt_y = smooth(args_piecewise, arr_piecewise_linear)
plt.plot(smt_x, smt_y, label='smoothed line')
plt.scatter(args, arr, color="red", label="experimental points")
plt.plot(X, func(X), label='function')
plt.grid()
plt.legend()
plt.show()

print('Cubic interpolation:')
error(smt_x, func(smt_x), smt_y, 'Square interpolation error')
print()

args_y = noised_func(args)
tck = scipy.interpolate.splrep(args, args_y)
sp_y = scipy.interpolate.splev(args, tck)
plt.grid()
plt.plot(X, func(X), label='function')
plt.plot(args, sp_y, "--", color="magenta", label='spline')
plt.legend()
plt.show()
print('Cubic spline:')
error(args, func(args), sp_y, 'Spline error')
print()

degree = [3, 4, 6, 12, 18]
print('Least square method:')
for i in degree:
    poly = search_square_polynomial_coefficients(X, noised_func(X), i)
    plt.scatter(args, arr, color="red", label="experimental points")
    plt.plot(X, func(X), label='function')
    polynomial_func = create_square_polynomial(poly, X)
    plt.plot(X, polynomial_func, color='green', label='Square interpolation degree = ' + str(i - 1))
    plt.legend()
    plt.grid()
    plt.show()
    error(X, func(X), polynomial_func, 'MNK error degree = ' + str(i - 1))
