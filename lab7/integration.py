import numpy as np
from prettytable import PrettyTable


def f(x):
    return np.sqrt(x) / (x + 1)


def F(x):
    return 2 * (np.sqrt(x) - np.arctan(np.sqrt(x)))


def get_n(method, f, a, b, eps=1e-3, p=2):
    n = 1
    t = 2 ** p - 1
    while True:
        int_1 = method(f, a, b, n)
        int_2 = method(f, a, b, 2 * n)
        if abs(int_1 - int_2) < eps:  # Правило Рунге / t
            break
        n *= 2
    return 2 * n


def trapeze_method(f, a, b, n):
    h = (b - a) / n
    s = sum((f(a + h * i) + f(a + h * (i + 1))) / 2 for i in range(n))
    return h * s


def simpson_method(f, a, b, n):
    h = (b - a) / (2 * n)
    return h / 3 * sum(
        [f(a + h * (i - 1)) + 4 * f(a + h * i) + f(a + h * (i + 1)) for i in range(1, 2 * n, 2)])


def newton_leibniz_method(a, b):
    return F(b) - F(a)


def main():
    a, b = 1, 4

    n = get_n(trapeze_method, f, a, b)
    h = (b - a) / n

    acc_val = newton_leibniz_method(a, b)

    int_tr1 = trapeze_method(f, a, b, n)
    int_tr2 = trapeze_method(f, a, b, n // 2)

    n_smp = get_n(simpson_method, f, a, b)
    int_smp1 = simpson_method(f, a, b, n=n)
    int_smp2 = simpson_method(f, a, b, n=n // 2)

    print('\nAccuracy table')
    table = PrettyTable()
    table.field_names = ['h', 'Newton-Leibniz', f'Trapeze(n={n})', '∆_1', f'Simpson(n={n_smp})', '∆_2']
    table.padding_width = 2
    table.add_row(['{:.4f}'.format(h), '{:.6f}'.format(acc_val), '{:.6f}'.format(int_tr1),
                   '{:.6e}'.format(abs(acc_val - int_tr1)), '{:.6f}'.format(int_smp1),
                   '{:.6e}'.format(abs(acc_val - int_smp1))])

    table.add_row(['{:.4f}'.format(2 * h), '', '{:.6f}'.format(int_tr2),
                   '{:.6e}'.format(abs(acc_val - int_tr2)), '{:.6f}'.format(int_smp2),
                   '{:.6e}'.format(abs(acc_val - int_smp2))])
    print(table)


if __name__ == '__main__':
    main()
