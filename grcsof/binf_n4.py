#!/usr/bin/env python3
"""k(1|4): the degenerate member a = infinity of the family h(a) = osp_a(4|2)
(h_1 = the cubics sigma(y_i)), binary operators of orders 1-4 at two pairs of
generic weights (Theorem 4.3.1 of the paper, the case a = infinity).

Usage:  python binf_n4.py
"""
from fractions import Fraction as Fr
from superN import add, smul
from k14family import ALG, CUB
from knclass import Classifier, label
from run_n4 import SIG
gens = [add(*[smul(SIG[i][j], CUB[j]) for j in range(4)]) for i in range(4)]
for l1, l2 in [(Fr(3,7), Fr(-5,11)), (Fr(2,13), Fr(-7,5))]:
    for N in [1,2,3,4]:
        C = Classifier(ALG, gens, l1, l2)
        ts, ns = C.solve(N, tmax=2)
        print(f"a=inf, l=({l1},{l2}), order {N}: dim = {len(ns)}", flush=True)
        if N == 2:
            for v in ns: print('   ', [(label(t, ALG.names), c) for t, c in zip(ts, v) if c != 0])
