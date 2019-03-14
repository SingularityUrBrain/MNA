from sympy import *

X = [0.351, 0.867, 1.315, 2.013, 2.859]
Y = [0.605, 0.218, 0.205, 1.157, 5.092]
N = 5


def fin_diff():
    y = Y.copy()
    for i in range(N - 1):
        for j in range(len(y) - 1):
            y[j] = y[j + 1] - y[j]
        y.pop()
        print(f'd{i + 1}y: {y}')


def div_diff():
    for i, f in zip(range(N - 1), f_gen()):
        print(f'f{i + 1}: {f}')


def f_gen():
    f = Y.copy()
    for i in range(N - 1):
        for j in range(len(f) - 1):
            f[j] = (f[j + 1] - f[j]) / (X[j + 1 + i] - X[j])
        f.pop()
        yield f


def newton():
    n = Y[0]
    for j, f in zip(range(N - 1), f_gen()):
        a = 1
        for k in range(j + 1):
            a *= (symbols('x') - X[k])
        n += f[0] * a
    return expand(n)


if __name__ == '__main__':
    print(f'x: {X}')
    print(f'y: {Y}\n')
    print('Finite differences:')
    fin_diff()
    print()
    print('Divided differences:')
    div_diff()
    n_pol = newton()
    print('\nN =', n_pol)
    print('N4(x1+x2) =', n_pol.subs(symbols('x'), X[1] + X[2]))
    plot(n_pol, line_color='y')
    plot(n_pol, xlim=[0, 4], ylim=[0, 10], line_color='y')
