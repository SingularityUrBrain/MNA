import numpy as np

A = []
eps = 1e-15
out = open('output_R.txt', 'w')

with open('input_4.txt', 'r') as input_f:
    n = int(input_f.readline())
    for line in input_f:
        temp = [float(x) for x in line.split()]
        A.append(temp)

A = np.array(A)
A_s = np.dot(A, np.transpose(A))  # symmetrization of the matrix A

out.write('A:\n {}\n'.format(A_s))

E, U = np.identity(n), np.identity(n)
k = 0
A_k = A_s

while True:
    k += 1
    temp = np.absolute(A_k - np.tril(A_k))  # upper triangle without diagonal
    t = np.sum([el ** 2 for el in temp])  # sum of squares of a non diagonal elements
    if t <= eps:
        break
    i, j = np.unravel_index(temp.argmax(), temp.shape)  # axis of a max element
    phi = np.arctan(2 * A_k[i][j] / (A_k[i][i] - A_k[j][j])) / 2
    U_k = np.identity(n)
    U_k[i][i], U_k[i][j], U_k[j][j], U_k[j][i] = np.cos(phi), -np.sin(phi), np.cos(phi), np.sin(phi)
    A_k = np.dot(np.dot(np.transpose(U_k), A_k), U_k)
    U = np.dot(U, U_k)  # U*U_k

out.write('Ak:\n {}\n'.format(A_k))
out.write('lambda:\n {}\n'.format(A_k.diagonal()))
out.write('U:\n {}\n'.format(U))
out.write('k:\n {}\n'.format(k))
out.write('eps:\n {}\n'.format(eps))
out.close()
