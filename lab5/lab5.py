import sympy as sp
import numpy as np
from matplotlib import pyplot
import eq_solve_methods.lineqs as lq
import eq_solve_methods.nonlineqs as nlq


def f(arg):
    return 0.1 * arg ** 2 - arg * np.log(arg)


def main():
    x = np.arange(0.1, 37, 0.1)
    y = 0.1 * x ** 2 - x * np.log(x)
    pyplot.plot(x, y)
    pyplot.plot([0, 40], [0, 0])
    pyplot.scatter(0, 0, c='w', linewidths=0.5, edgecolor='b')
    pyplot.grid(True)
    pyplot.show()  # ------------- 1

    eps = 1e-6
    x1, k1 = lq.solve_chord(f, 0.5, 5, eps)
    x2, k2 = lq.solve_chord(f, 40, 30, eps)
    k = k1 + k2
    print('Chord method:\n x = [{}, {}]\n k = {}'.format(x1, x2, k))  # ------------ 2

    x1, k1 = lq.solve_tangent(f, 5, eps)
    x2, k2 = lq.solve_tangent(f, 30, eps)
    k = k1 + k2
    print('Tangent method:\n x = [{}, {}]\n k = {}'.format(x1, x2, k))  # ------------ 3

    x, y = sp.symbols('x y')
    eq1 = sp.tan(x * y + 0.1) - 2 * x ** 2
    eq2 = 0.6 * x ** 2 + 2 * y ** 2 - 1
    p1 = sp.plotting.plot_implicit(eq1, (x, -1.5, 1.5), (y, -0.8, 0.8), show=False)
    p2 = sp.plotting.plot_implicit(eq2, (x, -1.5, 1.5), (y, -0.8, 0.8), show=False)
    p1.append(p2[0])
    p1.show()  # ------------- 4

    newton = nlq.solve_newton((0.3, 0.4), eps)
    print('\nNewton:\n x = {}\n k = {}'.format(newton[0], newton[1]))  # ------------- 7

    mod_newton = nlq.solve_newton((0.3, 0.4), eps, True)
    print('Modified Newton:\n x = {}\n k = {}'.format(mod_newton[0], mod_newton[1]))  # -- 8

    msi = nlq.solve_msi((0.5, 0.6), (0.1, 0.4), eps)
    print('MSI:\n x = {}\n k = {}'.format(msi[0], msi[1]))  # ----------- 6


if __name__ == '__main__':
    main()
