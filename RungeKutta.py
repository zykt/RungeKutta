import math
from collections import namedtuple
from numpy.linalg import norm


class System:
    """Evaluate right hand side of a diff equation system"""
    def __init__(self, right_hand_side):
        self.rhs = right_hand_side

    def apply(self, *args):
        return [f(*args) for f in self.rhs]


class RungeKuttaState:
    def __init__(self, x, ys, rhs: System, butcher_table: ButcherTable):
        self.x = x
        self.ys = ys
        self.rhs = rhs
        self.step = 2  # TODO: init step method
        self.table = butcher_table

    def initial_step(self, start, end, system: System, order):
        delta = (1 / max(start, end))**(s+1) + norm(system.apply(start))


    def correction_h(self, step):
        ks = []
        for (c, _as, b) in zip(*self.table):
            k_ys = [y + sum(a*k for (k, a) in zip(ks, _as)) for y in self.ys]
            ks.append([step * f(self.x + c*step, *k_ys) for f in self.fs])
        return ks

    def correction(self):
        return self.correction_h(self.step)

    def ys_hs(self, step):
        return [y + k for y, k in zip(self.ys, self.correction_h(step))]

    def next_ys(self):
        return [y + k for y, k in zip(self.ys, self.correction())]


class RungeKutta:
    def __init__(self, fs, init_values, start=0, end=math.pi):
        self.fs = fs
        self.init_values = init_values
        self.start = start
        self.end = end

    def solve(self):
        ys = self.init_values
        while True:
            state = RungeKuttaState(self.start, self.init_values, self.fs)
            break


ButcherTable = namedtuple('ButcherTable', ['c', 'a', 'b'])


def helper_table(c2):
    c = [0, c2]
    a = [[0, 0], [c2, 0]]
    b = [1 / (2*c2), 1 - 1 / (2*c2)]
    return ButcherTable(c, a, b)

if __name__ == '__main__':
    table = helper_table(1 / 12)