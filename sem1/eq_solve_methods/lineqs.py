def solve_chord(f, a, b, eps):
    x_k = b
    k = 0
    while True:
        k += 1
        x = x_k - f(x_k) * (a - x_k) / (f(a) - f(x_k))
        if abs(x - x_k) < eps:
            break
        x_k = x
    return x, k


def solve_tangent(f, init, eps):
    k = 0
    x_k = init - f(init) / d_fun(f, init)
    while True:
        k += 1
        x = x_k - f(x_k) / d_fun(f, x_k)
        if abs(x - x_k) < eps:
            break
        x_k = x
    return x, k


def d_fun(f, x):
    h = 1e-5
    return (f(x + h) - f(x)) / h
