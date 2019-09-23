from math import log

import numpy as np

A, b = [], []
eps = 1e-2
out = open('output_SI.txt', 'w')

with open('input_2.txt', 'r') as inp:
    n, m = map(int, inp.readline().split())
    for line in inp:
        temp = [float(x) for x in line.split()]
        b.append(temp.pop())
        A.append(temp)

A = np.array(A)
b = np.array(b)

out.write('A:\n {}\n'.format(A))
out.write('b:\n {}\n\n'.format(b))
if np.linalg.det(A) == 0:
    print('det_g == 0!')
    exit(1)

E = np.identity(len(A))
diag_A = np.diag(A)
c = b / diag_A  # c is ready
B = np.array(A)
for i in range(n):
    for j in range(m):
        B[i][j] /= diag_A[i]
B = E - B  # B is ready

norm_B = np.linalg.norm(B, np.inf)
if norm_B > 1:
    print('||B|| > 1, change matrix, please!')
    exit(1)
k = log(eps * (1 - norm_B), norm_B)  # count of iters in theory (k = log(eps * (1 - normB) / norm(c, 1), normB) - 1)

out.write('B:\n {}\n'.format(B))
out.write('c:\n {}\n\n'.format(c))
out.write('||B||: {:.3f} < 1\n\n'.format(norm_B))
out.write('k theoretic: {}\n\n'.format(int(k)))

x_k, x = b, np.zeros(n)  # if we take c instead b - count of iters will decrease (k=0 for this data)
k = 0

while True:
    k += 1
    x = np.dot(B, x_k) + c
    if abs(np.linalg.norm(x, np.inf) - np.linalg.norm(x_k, np.inf)) < eps:
        break
    x_k = x

out.write('x:\n {}\n\n'.format([round(sol, 3) for sol in x]))
out.write('k: {}\n\n'.format(k))
out.write('eps: {}\n\n'.format(eps))
out.close()
