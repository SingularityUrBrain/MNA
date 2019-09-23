from numpy import column_stack, array

import sem1.lab1.matrix_op as mop


def main():
    A, b = [], []
    with open('input_1.txt', 'r') as inp:
        for line in inp:
            temp = [float(x) for x in line.split()]
            b.append(temp.pop())
            A.append(temp)
    A = array(A)
    b = array(b)

    if mop.det(A) == 0:
        print('det_g == 0!')
        exit(1)

    xmatrix = column_stack((A, b))
    sols = mop.solve_gauss(xmatrix)
    mop.gauss(xmatrix, True)
    print('\nsolutions:', ['%.3f' % sol for sol in sols])
    abs_acc_x = mop.norm(mop.inverse(A)) * 0.001
    print('\nabs acc x = {}'.format(round(abs_acc_x, 5)))
    rel_acc_b = 0.001 / mop.norm(b)
    lim_rel_acc_x = round(mop.norm(mop.inverse(A)) * mop.norm(A) * rel_acc_b, 5)
    print('rel acc x <= {}'.format(lim_rel_acc_x))


if __name__ == '__main__':
    main()
