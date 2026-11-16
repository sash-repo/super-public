#!/usr/bin/env python3
"""the as-invariant operators between tensor fields of type V, n = 6.
Case (a) V = Lambda^2(id) = the 6-dimensional vector representation (L1),
case (b) V = id, id^* = the half-spinors (Sp, Sm),
case (c) V = S^2(id) = Lambda^3_xi (L3p; L3m is the other chirality)."""
import sys
from kasV import TYPES as T, usolve, bsolve, ulabel, blabel


def show(v, ts, lab):
    return "  ".join(f"{lab(t)}:{c}" for t, c in zip(ts, v) if c != 0)


def U(which, a, b, Ns):
    for N in Ns:
        ts, ns = usolve(which, T[a], T[b], N, quiet=True)
        print(f"{which}  unary  {a:4s} -> {b:4s}  order {N}: "
              f"{len(ts):5d} terms, dim = {len(ns)}", flush=True)
        for v in ns:
            print("      ", show(v, ts, ulabel), flush=True)


def B(which, a, b, c, Ns):
    for N in Ns:
        ts, ns = bsolve(which, T[a], T[b], T[c], N, quiet=True)
        print(f"{which}  binary {a:4s} x {b:4s} -> {c:4s}  order {N}: "
              f"{len(ts):5d} terms, dim = {len(ns)}", flush=True)
        for v in ns:
            print("      ", show(v, ts, blabel), flush=True)


if __name__ == '__main__':
    job = sys.argv[1]
    if job == 'a-unary':
        for N in (1, 2, 3, 4, 5, 6):
            U('as', 'F', 'L1', [N]); U('as', 'L1', 'F', [N]); U('as', 'L1', 'L1', [N])
    elif job == 'b-unary':
        for N in (1, 2, 3, 4):
            for p in [('F','Sp'),('Sp','F'),('Sp','Sp'),('Sm','Sm'),('Sp','Sm'),('Sm','Sp')]:
                U('as', p[0], p[1], [N])
    elif job == 'c-unary':
        for N in (1, 2, 3, 4):
            for p in [('F','L3p'),('L3p','F'),('F','L3m'),('L3m','F'),
                      ('L3p','L3p'),('L3m','L3m'),('L3p','L3m'),('L3m','L3p')]:
                U('as', p[0], p[1], [N])
    elif job == 'ac-cross':
        for N in (1, 2, 3, 4):
            for p in [('L1','L3p'),('L3p','L1'),('L1','L3m'),('L3m','L1'),
                      ('F','L2'),('L2','F'),('L1','L2'),('L2','L1')]:
                U('as', p[0], p[1], [N])
    elif job == 'a-binary':
        for N in (1, 2, 3, 4):
            B('as', 'F', 'F', 'L1', [N]); B('as', 'L1', 'F', 'F', [N])
            B('as', 'F', 'L1', 'F', [N]); B('as', 'L1', 'L1', 'F', [N])
    elif job == 'c-binary':
        for N in (1, 2, 3, 4):
            B('as', 'F', 'F', 'L3p', [N]); B('as', 'L3p', 'F', 'F', [N])
            B('as', 'F', 'F', 'L3m', [N]); B('as', 'L3m', 'F', 'F', [N])
    elif job == 'b-binary':
        for N in (1, 2, 3, 4):
            B('as', 'Sp', 'Sp', 'F', [N]); B('as', 'Sp', 'Sm', 'F', [N])
            B('as', 'Sm', 'Sm', 'F', [N]); B('as', 'Sm', 'Sp', 'F', [N])
