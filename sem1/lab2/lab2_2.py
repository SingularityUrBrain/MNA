from numpy import *
from numpy.linalg import *


A, b = [], []
eps = 1e-2
out = open('output_Z.txt', 'w')

with open('input_2.txt', 'r') as inp:
    n, m = map(int, inp.readline().split())
    for line in inp:
        temp = [float(x) for x in line.split()]
        b.append(temp.pop())
        A.append(temp)

A, b = array(A), array(b)

out.write('A:\n {}\n'.format(A))
out.write('b:\n {}\n\n'.format(b))
if det(A) == 0:
    print('det_g == 0!')
    exit(1)

E = identity(len(A))
D = diag(A)
LD = inv(tril(A))
U = triu(A) - E * D
x_k, x = array(b / D), zeros(n)
k = 0

out.write('x0:\n {}\n\n'.format(x_k))
out.write('(L+D)^(-1):\n {}\n'.format(LD))
out.write('U:\n {}\n\n'.format(U))

if norm(dot(LD, U), inf) > 1:
    print('||(L+D)^(-1)*U|| > 1, change matrix, please!')
    exit(1)
out.write('||(L+D)^(-1)*U||: {} < 1\n\n'.format(round(norm(dot(LD, U), inf), 3)))
while True:
    k += 1
    x = -dot(dot(LD, U), x_k) + dot(LD, b)
    if abs(norm(x_k, 1) - norm(x, 1)) < eps:
        break
    x_k = array(x)

out.write('x:\n {}\n\n'.format([round(sol, 3) for sol in x]))
out.write('k: {}\n\n'.format(k))
out.write('eps: {}\n\n'.format(eps))
out.close()
