import numpy as np
import sympy as sym

from sem1.lab6.LagrangeInterpPolynom import lagrange
from sem1.lab6.NewtonInterpPolynom import newton

X = [0.351, 0.867, 1.315, 2.013, 2.859]
Y = [0.605, 0.218, 0.205, 1.157, 5.092]
N = 5


def piecewise_linear_spline():
    a = []
    b = []
    phi = []
    for i in range(N - 1):
        a.append((Y[i + 1] - Y[i]) / (X[i + 1] - X[i]))
        b.append(Y[i] - a[i] * X[i])
        phi.append(a[i] * sym.symbols('x') + b[i])
    return phi


def piecewise_square_spline():
    a, b, c = [], [], []
    x = sym.symbols('x')
    phi = []
    for k in range(1, 3):
        M3 = np.array(
            [[X[2 * k] ** 2, X[2 * k], 1], [X[2 * k - 1] ** 2, X[2 * k - 1], 1], [X[2 * k - 2] ** 2, X[2 * k - 2], 1]])
        v3 = np.array([Y[2 * k], Y[2 * k - 1], Y[2 * k - 2]])
        sol = np.linalg.solve(M3, v3)
        a.append(sol[0])
        b.append(sol[1])
        c.append(sol[2])
        phi.append(a[k - 1] * x ** 2 + b[k - 1] * x + c[k - 1])
    return phi


def get_h(k):
    return X[k] - X[k - 1]


def get_l(k):
    return (Y[k] - Y[k - 1]) / get_h(k)


def delta1():
    return -1 / 2 * get_h(2) / (get_h(1) + get_h(2))


def lamb1():
    return 3 / 2 * (get_l(2) - get_l(1)) / (get_h(1) + get_h(2))


def get_delta(k):
    if k == 1:
        return delta1()
    return -get_h(k + 1) / (2 * get_h(k) + 2 * get_h(k + 1) + get_h(k) * get_delta(k - 1))


def get_lamb(k):
    if k == 1:
        return lamb1()
    return (2 * get_l(k + 1) - 3 * get_l(k) - get_h(k) * get_lamb(k - 1)) / (
            2 * get_h(k) + 2 * get_h(k + 1) + get_h(k) * get_delta(k - 1))


def get_a(k):
    return Y[k]


def get_c(k):
    if k == 0 or k == N - 1:
        return 0
    return get_delta(k) * get_c(k + 1) + get_lamb(k)


def get_b(k):
    return get_l(k) + 2 / 3 * get_c(k) * get_h(k) + 1 / 3 * get_h(k) * get_c(k - 1)


def get_d(k):
    return (get_c(k) - get_c(k - 1)) / (3 * get_h(k))


def piecewise_cubic_spline():
    s = []
    x = sym.symbols('x')
    for i in range(1, N):
        s.append(sym.expand(get_a(i) + get_b(i) * (x - X[i]) + get_c(i) * (x - X[i]) ** 2 + get_d(i) * (x - X[i]) ** 3))
    return s


if __name__ == '__main__':
    x = sym.symbols('x')

    s = piecewise_linear_spline()  # piecewise_linear_spline
    for piece in s:
        print('|', piece)
    print()
    f = sym.Piecewise((0, x < X[0]),
                      (s[0], sym.And(X[0] <= x, x < X[1])),
                      (s[1], sym.And(X[1] <= x, x < X[2])),
                      (s[2], sym.And(X[2] <= x, x < X[3])),
                      (s[3], sym.And(X[3] <= x, x <= X[4])),
                      (0, True), )
    p1 = sym.plot(f, xlim=[0, 4])

    s2 = piecewise_square_spline()  # piecewise_square_spline
    for piece in s2:
        print('|', piece)
    print()
    f2 = sym.Piecewise((0, x < X[0]),
                       (s2[0], sym.And(X[0] <= x, x < X[2])),
                       (s2[1], sym.And(X[2] <= x, x < X[4])),
                       (0, True))
    p2 = sym.plot(f2, xlim=[0, 4], line_color='r')

    p1.append(p2[0])
    p1.show()

    s3 = piecewise_cubic_spline()  # piecewise_cubic_spline
    for cub_eq in s3:
        print('|', cub_eq)
    f = sym.Piecewise((0, x < X[0]),
                      (s3[0], sym.And(X[0] <= x, x < X[1])),
                      (s3[1], sym.And(X[1] <= x, x < X[2])),
                      (s3[2], sym.And(X[2] <= x, x < X[3])),
                      (s3[3], sym.And(X[3] <= x, x < X[4])),
                      (0, True))
    p3 = sym.plot(f, xlim=[0, 4], line_color='violet')
    p1.append(p3[0])
    p1.show()

    p4 = sym.plot(newton(), xlim=[0, 4], ylim=[0, 10], line_color='y', show=False)
    p5 = sym.plot(lagrange(), xlim=[0, 4], ylim=[0, 10], line_color='g', show=False)
    p1.extend([p4[0], p5[0]])
    p1.ylim = [0, 10]
    p1.show()
