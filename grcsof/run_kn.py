#!/usr/bin/env python3
"""Driver: dimension of the space of invariant binary operators of order N
for k(1|n), n = 1,2,3,4."""
import sys
from fractions import Fraction as Fr
from superN import Contact, mono
from knclass import Classifier, osp_gens, label, check_neg

ALGS = {
    1: Contact(1, [[1]], names=['th']),
    2: Contact(2, [[0, 1], [1, 0]], names=['xi', 'eta']),
    '2d': Contact(2, [[1, 0], [0, 1]], names=['th1', 'th2']),
    3: Contact(3, [[1, 0, 0], [0, 0, 1], [0, 1, 0]], names=['th', 'xi', 'eta']),
    '3d': Contact(3, [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  names=['th1', 'th2', 'th3']),
    4: Contact(4, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
               names=['xi1', 'eta1', 'xi2', 'eta2']),
}


def run(key, Ns, lam1, lam2, tmax=None, gens=None, verbose=True):
    alg = ALGS[key]
    gens = gens if gens is not None else osp_gens(alg)
    out = {}
    for N in Ns:
        C = Classifier(alg, gens, lam1, lam2)
        ts, ns = C.solve(N, tmax=tmax, verbose=verbose)
        out[N] = (ts, ns)
    return out


if __name__ == '__main__':
    key = sys.argv[1]
    key = int(key) if key.isdigit() else key
    Ns = [int(x) for x in sys.argv[2].split(',')]
    if len(sys.argv) > 3 and sys.argv[3] == 'sym':
        from sympy import symbols
        l1, l2 = symbols('l1 l2')
    else:
        l1, l2 = Fr(3, 7), Fr(-5, 11)      # generic rational weights
    tmax = int(sys.argv[4]) if len(sys.argv) > 4 else None
    print(f"k(1|{key}), weights {l1}, {l2}")
    run(key, Ns, l1, l2, tmax=tmax)
