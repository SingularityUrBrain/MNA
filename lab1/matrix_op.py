import numpy as np


def gauss_jordan(matrix):
    n = matrix.shape[0]  # rows
    m = matrix.shape[1]  # cols

    # direct Gauss's way
    a = gauss(matrix)
    # ones on the diagonal
    for i in range(n):
        coeff = a[i][i]
        for j in range(m):
            a[i][j] /= coeff
    # inverse
    for step in range(n - 1, -1, -1):
        for i in range(step - 1, -1, -1):
            if a[step][step] == 0:
                return None
            coeff = a[i][step] / a[step][step]
            for j in range(step, m):
                a[i][j] -= a[step][j] * coeff

    return a


def gauss(matrix, show=False):
    n = matrix.shape[0]
    m = matrix.shape[1]
    a = matrix.copy()
    for step in range(n - 1):
        for i in range(step + 1, n):
            if a[step][step] == 0:
                if show:
                    raise Exception('Gauss method is not available for this matrix')
                return None
            coeff = a[i][step] / a[step][step]
            for j in range(step, m):
                a[i][j] -= a[step][j] * coeff
        if show:
            print('\nstep {}:\n'.format(step + 1), a)
    return a


def det_g(matrix):
    n = matrix.shape[0]
    a = gauss(matrix)
    if a is None or n != matrix.shape[1]:
        return None
    det = 1
    for i in range(n):
        det *= a[i][i]
    return det


def solve_gauss(xmatrix):
    n = xmatrix.shape[0]
    a = gauss(xmatrix)
    if a is None or (xmatrix.shape[1] - xmatrix.shape[0]) != 1:
        raise Exception('Gauss method is not available for this matrix')
    sols = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        sols[i] = a[i][n] / a[i][i]
        for k in range(i - 1, -1, -1):
            a[k][n] -= a[k][i] * sols[i]
    return sols


def minor(arr, i, j):
    minor = [list(row[:j]) + list(row[j + 1:]) for row in (list(arr[:i]) + list(arr[i + 1:]))]
    return minor


def det(m):
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * det(minor(m, 0, c))
    return determinant


def inverse(matrix):
    m = list(matrix)
    determinant = det(m)
    if det(m) == 0:
        print('det_g = 0!')
        exit()

    if len(m) == 2:
        return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                [-1 * m[1][0] / determinant, m[0][0] / determinant]]

    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor_ls = minor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * det(minor_ls))
        cofactors.append(cofactorRow)
    cofactors = list(map(list, zip(*(cofactors))))
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return np.array(cofactors)


def norm(matrix):
    if len(matrix) == 1:
        return max(matrix[0])
    else:
        return max([np.sum(np.abs(i)) for i in matrix])
