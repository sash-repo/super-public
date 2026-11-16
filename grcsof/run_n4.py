#!/usr/bin/env python3
"""k(1|4): invariant binary operators for the family h(a) = osp_a(4|2),
h_1(a) = <t y_i + a sigma(y_i)>, see k14family.py."""
import sys
from fractions import Fraction as Fr
from superN import mono, add, smul
from k14family import ALG, TY, CUB
from knclass import Classifier, label

SIG = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]]


def gens(a):
    return [add(TY[i], *[smul(a * SIG[i][j], CUB[j]) for j in range(4)])
            for i in range(4)]


if __name__ == '__main__':
    Ns = [int(x) for x in sys.argv[1].split(',')]
    avals = [Fr(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [Fr(0)]
    l1, l2 = Fr(3, 7), Fr(-5, 11)
    if len(sys.argv) > 3 and sys.argv[3] == 'sym':
        from sympy import symbols
        l1, l2 = symbols('l1 l2')
    tmax = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    for a in avals:
        for N in Ns:
            C = Classifier(ALG, gens(a), l1, l2)
            ts, ns = C.solve(N, tmax=tmax)
            print(f"a = {a}, order {N}: {len(ts)} terms, dim = {len(ns)}")
            for v in ns:
                from sympy import factor, simplify, S
                items = [(label(t, ALG.names), c) for t, c in zip(ts, v) if c != 0]
                for nm, c in items:
                    print(f"     {nm} : {factor(simplify(S(c)))}")
            sys.stdout.flush()
