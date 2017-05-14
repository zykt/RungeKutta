import math
import numpy as np
from numpy.linalg import norm
from RungeKutta import System, ButcherTable, init_step, correction


log = {
    "task2": {"x": [], "true_error": [], "h": 0},
    "task3.1": {"x": [], "h": []},
    "task3.2": {"x": [], "estimate_error": [], "true_error": []},
    "task3.3": {"tolerance": [], "rhs_evaluations": 0}
}


#####################
# const step solver #
#####################


def const_step_solve(system: System, init_values, table: ButcherTable, start=0, end=math.pi, order=2, tolerance=10 ** -6):
    step = init_step(system, init_values, start, end, order, tolerance)

    ys_doublestep = _const_step_solver(start, np.array(init_values), step * 2, system, table, end)

    while True:
        ys_step = _const_step_solver(start, np.array(init_values), step, system, table, end)

        global_error = norm(ys_step - ys_doublestep) / (2**order - 1)

        if global_error < tolerance:
            break
        else:
            ys_doublestep = ys_step
            step /= 2

    return ys_step, global_error


def _const_step_solver(x, ys, step, system: System, table: ButcherTable, end):
    while True:
        if step + x >= end:
            step = end - x
            x = end
        else:
            x += step
        ys += correction(x, ys, system, step, table)
        if x >= end:
            break
    return ys

###################
# var step solver #
###################


def var_step_solve(system: System, init_values, table: ButcherTable, start=0, end=math.pi, order=2, error=10**-4):
    step = init_step(system, init_values, start, end, order, error)

    ys_step = _var_step_solver(start, np.array(init_values), step, system, table, end, order, error)
    ys_doublestep = _var_step_solver(start, np.array(init_values), step * 2, system, table, end, order, error)

    error = norm(ys_step - ys_doublestep) / (2**order - 1)

    return ys_step, error


def _var_step_solver(x, ys, step, system: System, table: ButcherTable, end, order, error):
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

        next_ys = ys + correction(x, ys, system, step, table)
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
    def make_linear_func(A=0, B=0):
        return lambda x, y1, y2: B*y1 + A*y2

    def helper_table(c2):
        """Compute values for Butcher Table from c2"""
        c = [0, c2]
        a = [[0, 0], [c2, 0]]
        b = [1 / (2 * c2), 1 - 1 / (2 * c2)]
        return ButcherTable(c, a, b)


    def classic_runge_kutta_method():
        c = [0, 0.5, 0.5, 1]
        a = [[0, 0, 0, 0],
             [0.5, 0, 0, 0],
             [0, 0.5, 0, 0],
             [0, 0, 1, 0]]
        b = [1/6, 1/3, 1/3, 1/6]
        return ButcherTable(c, a, b)


    A, B = 1/12, 1/15

    dy1_by_dx = make_linear_func(A=A)
    dy2_by_dx = make_linear_func(B=-B)
    sys = System([dy1_by_dx, dy2_by_dx])

    init = [B * math.pi, A * math.pi]

    table = classic_runge_kutta_method()

    print('#'*50)
    print('Const step result:')
    print(const_step_solve(sys, init, table))
    print('#'*50)

    print(' '*50)

    print('#'*50)
    print('Var step result:')
    print(var_step_solve(sys, init, table))
    print('#'*50)
