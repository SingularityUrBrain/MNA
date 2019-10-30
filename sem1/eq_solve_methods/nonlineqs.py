import numpy as np
import sympy


def jacobi(x_v: list):
    n = len(x_v)
    x, y = sympy.symbols('x y')
    y1 = sympy.tan(x * y + 0.1) - 2 * x ** 2
    y2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
    eqs = [y1, y2]
    x_v_s = [x, y]
    J = np.identity(n)

    for i in range(n):
        for j in range(n):
            J[i][j] = sympy.diff(eqs[i], x_v_s[j]).subs(x_v_s[0], x_v[0]).subs(x_v_s[1], x_v[1])
    return J


def solve_newton(init: tuple, eps, mod=False):
    x, y = init
    k = 0
    if not mod:
        while True:
            k += 1
            y1 = sympy.tan(x * y + 0.1) - 2 * x ** 2
            y2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
            J = jacobi([x, y])
            x_sol = np.array([x, y]) - np.dot(np.linalg.inv(J), np.array([y1, y2]))
            if abs(x_sol[0] - x) < eps:
                break
            x, y = x_sol[0], x_sol[1]
    else:
        J = jacobi([x, y])
        while True:
            k += 1
            y1 = sympy.tan(x * y + 0.1) - 2 * x ** 2
            y2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
            x_sol = np.array([x, y]) - np.dot(np.linalg.inv(J), np.array([y1, y2]))
            if abs(x_sol[0] - x) < eps:
                break
            x, y = x_sol[0], x_sol[1]
    return x_sol, k


def solve_msi(init: tuple, end: tuple, eps):
    n = len(init)
    y1 = lambda x, y: sympy.tan(x * y + 0.1) - 2 * x ** 2
    y2 = lambda x, y: 0.6 * x ** 2 + 2 * y ** 2 - 1
    f_v = [y1, y2]
    x_sol = [0, 0]
    a = init
    x_k = end
    k = 0
    while True:
        k += 1
        for i in range(n):
            x_sol[i] = x_k[i] - f_v[i](x_k[0], x_k[1]) * (a[i] - x_k[i]) / (f_v[i](a[0], a[1]) - f_v[i](x_k[0], x_k[1]))
        if abs(x_sol[0] - x_k[0]) < eps:
            break
        x_k = x_sol.copy()
    return x_sol, k
