from numpy import *
from math import copysign, sqrt, fabs, fsum
from lab1.matrix_op import gauss_jordan


def print_system(file):
    for i in range(n):
        file.write(' |')
        for j in range(m):
            file.write('{:.3f}*x{}'.format(A[i][j], j + 1))
            file.write(' + ') if j != m - 1 else None
        file.write(f' = {b[i]}\n')
    file.write('\nA:\n {}\n'.format(A))
    file.write('b:\n {}\n\n'.format(b))


A, b = [], []
out = open('output_IR.txt', 'w')
with open('input_3.txt', 'r') as inp:
    n, m = map(int, inp.readline().split())
    for line in inp:
        temp = [float(x) for x in line.split()]
        b.append(temp.pop())
        A.append(temp)
A, b = array(A), array(b)
out.write('System:\n')
print_system(out)

A_s, b_s = dot(A, transpose(A)), dot(b, transpose(A))  # symmetrization of the matrix a, b ----- 1
out.write('A*:\n {}\n'.format(A_s))
out.write('b*:\n {}\n'.format(b_s))

if linalg.det(A) == 0:  # check det_g!=0
    print('det_g == 0!')
    exit(1)

s, d = zeros((n, m)), zeros(n)
s[0][0] = sqrt(fabs(A_s[0][0]))
d[0] = copysign(1, A_s[0][0])

for i in range(n):  # find S & D using formula
    val = A_s[i][i] - fsum([(s[k][i] ** 2) * d[k] for k in range(i)])
    d[i] = copysign(1, val)
    s[i][i] = sqrt(fabs(val))
    for j in range(i, n):
        s[i][j] = (A_s[i][j] - fsum([s[k][i] * d[k] * s[k][j] for k in range(i)])) / (s[i][i] * d[i])
out.write('S:\n {}\n'.format(s))
out.write('D:\n {}\n'.format(d))

y = zeros(n)
y[0] = b_s[0] / s[0][0]  # find y
for i in range(1, n):
    y[i] = (b_s[i] - fsum([s[k][i] * y[k] for k in range(i)])) / s[i][i]
out.write('y:\n {}\n'.format(y))

x = zeros(n)
x[n - 1] = y[n - 1] / s[n - 1][n - 1]                  # find x ---------- 2
for i in range(n - 2, -1, -1):
    x[i] = (y[i] - fsum([s[i][k] * x[k] for k in range(i + 1, n)])) / s[i][i]
out.write('\nx:\n {}\nx(numpy):\n {}\n\n'.format([round(sol, 3) for sol in x], linalg.solve(A, b)))

det = prod([(s[i][i] ** 2) * d[i] for i in range(n)])  # find determinant ----------- 3
out.write('det_g: {:>19.3f}\ndet_g(numpy): {:>12.3f}\n'.format(sqrt(det), linalg.det(A)))

x_A = column_stack((A, identity(n)))    # A|E
B = gauss_jordan(x_A)                   # E|A^-1
B = array([row[n:] for row in B])                      # find A^-1 ------------ 4
out.write('\nA^-1:\n {}\nA^-1(numpy):\n {}\n'.format(B, linalg.inv(A)))
out.close()
