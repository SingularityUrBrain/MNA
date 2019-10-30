import numpy as np


def main():
    A, b = [], []
    with open('input_1.txt', 'r') as inp:
        for line in inp:
            temp = [float(x) for x in line.split()]
            b.append(temp.pop())
            A.append(temp)
    A = np.array(A)
    b = np.array(b)
    if np.linalg.det(A) == 0:
        print('det == 0!')
        exit(1)

    sols = np.linalg.solve(A, b)
    print('\nsolutions:', ['%.3f' % sol for sol in sols])
    A_inv = np.linalg.inv(A)
    print('Inversion matrix:\n', A_inv)
    abs_acc_x = np.linalg.norm(A_inv, ord=np.inf) * 0.001
    print('\nabs acc x = {}'.format(round(abs_acc_x, 5)))
    rel_acc_b = 0.001 / np.linalg.norm(b, ord=np.inf)
    lim_rel_acc_x = round(np.linalg.norm(A_inv, ord=np.inf) * np.linalg.norm(A, ord=np.inf) * rel_acc_b, 5)
    print('rel acc x <= {}'.format(lim_rel_acc_x))


if __name__ == '__main__':
    main()
