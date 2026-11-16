#!/usr/bin/env python3
"""k(1|5): the gap of Section "What is not done here".

The pair is i(osp(5|2)) subset k(1|5), in the same conventions as for n <= 4:
five odd indeterminates with the Gram matrix

    B = <th> (+) hyperbolic(xi1,eta1) (+) hyperbolic(xi2,eta2),

i.e. alpha_1 = dt + th d(th) + xi1 d(eta1) + eta1 d(xi1) + xi2 d(eta2)
                                + eta2 d(xi2).

Nothing in superN.py / knclass.py is specific to n <= 4, so the only new datum
is the Gram matrix; everything else is the machinery already validated on
n = 1,2,3,4 (see check_superN.py, verify_kn.py).
"""
import sys
from fractions import Fraction as Fr
from superN import Contact
from knclass import Classifier, osp_gens, label

B5 = [[1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0],
      [0, 1, 0, 0, 0],
      [0, 0, 0, 0, 1],
      [0, 0, 0, 1, 0]]

ALG = Contact(5, B5, names=['th', 'xi1', 'eta1', 'xi2', 'eta2'])


def run(Ns, l1, l2, tmax=2, show=True):
    out = {}
    for N in Ns:
        C = Classifier(ALG, osp_gens(ALG), l1, l2)
        ts, ns = C.solve(N, tmax=tmax)
        out[N] = (ts, ns)
        print(f"order {N}: {len(ts)} terms, dim = {len(ns)}")
        sys.stdout.flush()
        if show:
            for v in ns:
                print("  solution:")
                for t, c in zip(ts, v):
                    if c != 0:
                        print(f"     {label(t, ALG.names)} : {c}")
                sys.stdout.flush()
    return out


if __name__ == '__main__':
    Ns = [int(x) for x in sys.argv[1].split(',')]
    if len(sys.argv) > 2 and sys.argv[2] == 'sym':
        from sympy import symbols
        l1, l2 = symbols('l1 l2')
    else:
        l1, l2 = Fr(3, 7), Fr(-5, 11)          # generic rational weights
    tmax = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    print(f"k(1|5), i(osp(5|2)), weights {l1}, {l2}, tmax={tmax}")
    run(Ns, l1, l2, tmax=tmax)
