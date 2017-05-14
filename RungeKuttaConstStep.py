import math

import numpy as np
from numpy.linalg import norm

from RungeKutta import System, ButcherTable, init_step, correction


def solve(system: System, init_values, table: ButcherTable, start=0, end=math.pi, order=2, tolerance=10 ** -6):
    step = init_step(system, init_values, start, end, order, tolerance)

    ys_doublestep = solver(start, np.array(init_values), step*2, system, table, end)

    while True:
        ys_step = solver(start, np.array(init_values), step, system, table, end)

        global_error = norm(ys_step - ys_doublestep) / (2**order - 1)

        if global_error < tolerance:
            break
        else:
            ys_doublestep = ys_step
            step /= 2

    return ys_step, global_error


def solver(x, ys, step, system: System, table: ButcherTable, end):
    while True:
        if step + x >= end:
            step = end - x
            x = end
        else:
            x += step
        ys += np.array(correction(x, ys, system, step, table))
        if x >= end:
            break
    return ys


def helper_table(c2):
    """Compute values for Butcher Table from c2"""
    c = [0, c2]
    a = [[0, 0], [c2, 0]]
    b = [1 / (2*c2), 1 - 1 / (2*c2)]
    return ButcherTable(c, a, b)


if __name__ == '__main__':
    def make_linear_func(A=0, B=0): return lambda x, y1, y2: B*y1 + A*y2

    A, B = 1/12, 1/15

    dy1_by_dx = make_linear_func(A=A)
    dy2_by_dx = make_linear_func(B=-B)
    sys = System([dy1_by_dx, dy2_by_dx])

    init = [B * math.pi, A * math.pi]

    table = helper_table(1 / 12)

    print('#'*25)
    print('Result:')
    print(solve(sys, init, table))