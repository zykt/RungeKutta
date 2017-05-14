import math
import numpy as np
from collections import namedtuple
from numpy.linalg import norm
from itertools import zip_longest
from RungeKutta import System, ButcherTable, init_step, correction


def solve(system: System, init_values, table: ButcherTable, start=0, end=math.pi, order=2, error=10**-6):
    step = init_step(system, init_values, start, end, order, error)

    ys_step = solver(start, np.array(init_values), step, system, table, end, order, error)
    ys_doublestep = solver(start, np.array(init_values), step*2, system, table, end, order, error)

    error = norm(ys_step - ys_doublestep) / (2**order - 1)

    return ys_step, error


def solver(x, ys, step, system: System, table: ButcherTable, end, order, error):
    def take_doublestep(_x, _ys, _step):
        _ys += np.array(correction(_x, _ys, system, _step, table))
        _x += _step
        _ys += np.array(correction(_x, _ys, system, _step, table))
        return _ys

    while True:
        if step + x >= end:
            step = end - x
            x = end
        else:
            x += step

        next_ys = ys + np.array(correction(x, ys, system, step, table))
        ys_doublestep = take_doublestep(x, ys, step/2)
        local_error = norm(ys_doublestep - next_ys) / (1 - 2**(-order))

        if local_error > error * 2**order:
            step /= 2
        elif local_error > error:
            step /= 2
            ys = ys_doublestep
        elif local_error > error / 2**(order+1):
            ys = next_ys
        else:
            step *= 2
            ys = next_ys

        if x >= end:
            break
    return ys


if __name__ == '__main__':
    def make_linear_func(A=0, B=0): return lambda x, y1, y2: B*y1 + A*y2

    def helper_table(c2):
        """Compute values for Butcher Table from c2"""
        c = [0, c2]
        a = [[0, 0], [c2, 0]]
        b = [1 / (2 * c2), 1 - 1 / (2 * c2)]
        return ButcherTable(c, a, b)

    A, B = 1/12, 1/15

    dy1_by_dx = make_linear_func(A=A)
    dy2_by_dx = make_linear_func(B=-B)
    sys = System([dy1_by_dx, dy2_by_dx])

    init = [B * math.pi, A * math.pi]

    table = helper_table(1 / 12)

    print('#'*25)
    print('Result:')
    print(solve(sys, init, table))
