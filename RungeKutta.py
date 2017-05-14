from collections import namedtuple
from itertools import zip_longest

import numpy as np
from numpy.linalg import norm


class System:
    """Evaluate right hand side of a diff equation system"""
    def __init__(self, right_hand_side):
        self.rhs = right_hand_side

    def apply(self, *args):
        # print(args)
        return [f(*args) for f in self.rhs]


ButcherTable = namedtuple('ButcherTable', ['c', 'a', 'b'])


def init_step(system: System, init_values, start, end,  order, error):
    delta = (1 / max(abs(start), abs(end)))**(order+1) + norm(system.apply(start, *init_values))
    return (error / delta)**(1 / (order+1))


def correction(x, ys, system: System, step, table: ButcherTable):
    """Compute correction as h*f(x+c*h, y+sum(c_i*k_i))"""
    kss = []
    for c, a, b in zip(*table):
        ks = [[a*k for k in ks] for a, ks in zip(a, kss)]
        ks = np.sum(ks, axis=0) if ks else []
        ys = [y + k for y, k in zip_longest(ys, ks, fillvalue=0)]
        sys = system.apply(x + c*step, *ys)
        kss.append([b * step * f for f in sys])
    return np.sum(kss, axis=0)