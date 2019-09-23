import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def f(x, y):
    return (-(5 * x ** 2 + 3) * y ** 3 - 12 * y) / (8 * x)


def y(x):
    return 2 / np.sqrt(10 * x ** 3 - 5 * x ** 2 - 1)


def get_n(method, func, a, b, f0, p, eps=1e-4):
    n = 1
    t = 2 ** p - 1
    while True:
        _, y1 = method(func, a, b, f0, n)
        _, y2 = method(func, a, b, f0, 2 * n)
        if abs(y1[-1] - y2[-1]) < eps:  # /t Runge's rule
            break
        n *= 2

    return n * 2


def runge_kutta_method(func, a, b, f0, n):
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = a
    y[0] = f0
    for k in range(n):
        f1 = func(x[k], y[k])
        f2 = func(x[k] + h / 2, y[k] + f1 * h / 2)
        f3 = func(x[k] + h / 2, y[k] + f2 * h / 2)
        f4 = func(x[k] + h, y[k] + h * f3)
        y[k + 1] = y[k] + h / 6 * (f1 + f4 + 2 * (f2 + f3))
        x[k + 1] = x[k] + h

    return x, y


def adams_method(func, a, b, f0, n):
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = a
    y[0] = f0
    x[1] = x[0] + h
    y[1] = y[0] + h * func(x[0], y[0])
    for k in range(1, n):
        predictor = y[k] + h / 2 * (3 * func(x[k], y[k]) - func(x[k - 1], y[k - 1]))
        x[k + 1] = x[k] + h
        y[k + 1] = y[k] + h / 2 * (func(x[k], y[k]) + func(x[k + 1], predictor))

    return x, y


def euler_method(func, a, b, f0, n):
    h = (b - a) / n
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    x[0] = a
    y[0] = f0

    for i in range(n):
        y[i + 1] = y[i] + h * func(x[i], y[i])
        x[i + 1] = x[i] + h

    return x, y


def plot_one_show(title, *args):
    """args = x,y,label, x1,y1,label2 ..."""

    print(len(args))
    for i in range(0, len(args), 3):
        plt.plot(args[i], args[i + 1], label=args[i + 2])
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def print_table(x, y, y2):
    table = PrettyTable()
    table.padding_width = 3
    table.field_names = ['x_i', 'y_i', '~y_i', '|y_i - ~y_i|']
    # table.align = 'l'
    for i in range(len(x)):
        table.add_row(
            ['{:.4f}'.format(x[i]), '{:.6f}'.format(y[i]), '{:.6f}'.format(y2[i // 2]) if i % 2 == 0 else '',
             '{:.6e}'.format(abs(y[i] - y2[i // 2])) if i % 2 == 0 else ''])
    print(table)


def main():
    a, b = 1, 3
    f0 = 1

    n = get_n(runge_kutta_method, f, a, b, f0, p=4)
    h = (b - a) / n

    # Runge-Kutta method
    rk_x, rk_y = runge_kutta_method(f, a, b, f0, n)
    rk_x2, rk_y2 = runge_kutta_method(f, a, b, f0, n // 2)
    print(' n = {}\n h = {}\n 2h = {}\n'.format(n, h, 2 * h))
    print(f'Runge-Kutta method (n = {n})')
    print_table(rk_x, rk_y, rk_y2)

    # Adams method
    n_a = get_n(adams_method, f, a, b, f0, p=2)
    a_x, a_y = adams_method(f, a, b, f0, n)
    a_x2, a_y2 = adams_method(f, a, b, f0, n // 2)
    print()
    print(f'Adams method (n = {n_a})')
    print_table(a_x, a_y, a_y2)

    # Euler method
    n_e = get_n(euler_method, f, a, b, f0, p=2)
    e_x, e_y = euler_method(f, a, b, f0, n)
    e_x2, e_y2 = euler_method(f, a, b, f0, n // 2)
    print()
    print(f'Euler method (n = {n_e})')
    print_table(e_x, e_y, e_y2)

    # Accuracy table
    print('\nAccuracy table')
    table = PrettyTable()
    table.field_names = ['x_i', 'Exact', 'Runge-Kutta', '∆_i1', 'Adams', '∆_i2']
    table.padding_width = 2
    x = a
    for i in range(n):
        sol = y(x)
        table.add_row(['{:.4f}'.format(x), '{:.6f}'.format(sol), '{:.6f}'.format(rk_y[i]),
                       '{:.6e}'.format(abs(sol - rk_y[i])), '{:.6f}'.format(a_y[i]),
                       '{:.6e}'.format(abs(sol - a_y[i]))])
        x += h
    print(table)

    # Graphics
    plot_one_show('Runge-Kutta', rk_x, rk_y, 'h', rk_x2, rk_y2, '2h')
    plot_one_show('Adams', a_x, a_y, 'h', a_x2, a_y2, '2h')
    plot_one_show('Euler', e_x, e_y, 'h', e_x2, e_y2, '2h')

    x = np.linspace(a, b)
    plot_one_show('', rk_x, rk_y, 'Runge-Kutta', a_x, a_y, 'Adams', e_x, e_y, 'Euler', x, y(x), 'original')


if __name__ == '__main__':
    main()
